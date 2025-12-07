#pragma once
#include <GigaLearnCPP/Framework.h>
#include <RLGymCPP/BasicTypes/Lists.h>

// Include torch (full C++ API for torch::Tensor plus ATen utilities)
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/lazy/core/ops/utils.h>
#if __has_include(<torch/cuda.h>)
#include <torch/cuda.h>
#else
namespace torch {
    namespace cuda {
        inline bool is_available() { return false; }
        inline int device_count() { return 0; }
    }
}
#endif

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#endif

#define RG_NO_GRAD torch::NoGradGuard _noGradGuard

// OPTIMISATION: inference_mode est plus rapide que NoGradGuard car il désactive aussi le version tracking
#define RG_INFERENCE_MODE torch::InferenceMode _inferenceMode

#define RG_AUTOCAST_ON() { \
at::autocast::set_enabled(true); \
at::autocast::set_autocast_gpu_dtype(torch::kBFloat16); \
at::autocast::set_autocast_cpu_dtype(torch::kFloat); \
}

#define RG_AUTOCAST_OFF() { \
at::autocast::clear_cache(); \
at::autocast::set_enabled(false); \
}

#define RG_HALFPERC_TYPE torch::ScalarType::BFloat16

namespace GGL {

	// OPTIMISATION: Cache pour les options de tenseurs fréquemment utilisées
	struct TensorOptionsCache {
		static torch::TensorOptions& Float() {
			static torch::TensorOptions opts = torch::TensorOptions()
				.dtype(torch::kFloat32)
				.device(torch::kCPU);
			return opts;
		}
		
		static torch::TensorOptions& Int64() {
			static torch::TensorOptions opts = torch::TensorOptions()
				.dtype(torch::kInt64)
				.device(torch::kCPU);
			return opts;
		}
		
		static torch::TensorOptions& Int32() {
			static torch::TensorOptions opts = torch::TensorOptions()
				.dtype(torch::kInt32)
				.device(torch::kCPU);
			return opts;
		}
		
		static torch::TensorOptions& Uint8() {
			static torch::TensorOptions opts = torch::TensorOptions()
				.dtype(torch::kUInt8)
				.device(torch::kCPU);
			return opts;
		}
		
		static torch::TensorOptions& Int8() {
			static torch::TensorOptions opts = torch::TensorOptions()
				.dtype(torch::kInt8)
				.device(torch::kCPU);
			return opts;
		}
	};

	// OPTIMISATION: Fonction template optimisée pour obtenir les options cachées
	template<typename T>
	inline const torch::TensorOptions& GetCachedOptions() {
		if constexpr (std::is_same_v<T, float>) {
			return TensorOptionsCache::Float();
		} else if constexpr (std::is_same_v<T, int64_t>) {
			return TensorOptionsCache::Int64();
		} else if constexpr (std::is_same_v<T, int32_t>) {
			return TensorOptionsCache::Int32();
		} else if constexpr (std::is_same_v<T, uint8_t>) {
			return TensorOptionsCache::Uint8();
		} else if constexpr (std::is_same_v<T, int8_t>) {
			return TensorOptionsCache::Int8();
		} else {
			static torch::TensorOptions opts = torch::TensorOptions()
				.dtype(torch::CppTypeToScalarType<T>())
				.device(torch::kCPU);
			return opts;
		}
	}

	// OPTIMISATION: Fast memcpy pour les conversions
	template <typename T>
	inline torch::Tensor DIMLIST2_TO_TENSOR(const RLGC::DimList2<T>& list) {
		if (list.numel == 0) {
			return torch::empty({0, 0}, GetCachedOptions<T>());
		}
		
		auto t = torch::empty({(int64_t)list.size[0], (int64_t)list.size[1]}, GetCachedOptions<T>());
		std::memcpy(t.data_ptr<T>(), list.data.data(), list.numel * sizeof(T));
		return t;
	}

	template <typename T>
	inline torch::Tensor VectorToTensor(const std::vector<T>& data, const std::vector<int64_t>& shape) {
		if (data.empty()) {
			return torch::empty(shape, GetCachedOptions<T>());
		}

		int64_t expected = 1;
		for (auto s : shape) expected *= s;
		if (expected != (int64_t)data.size()) {
			return torch::tensor(data);
		}

		auto t = torch::empty(shape, GetCachedOptions<T>());
		std::memcpy(t.data_ptr<T>(), data.data(), data.size() * sizeof(T));
		return t;
	}

	// OPTIMISATION: Version qui réutilise un tensor existant pour éviter les allocations
	template <typename T>
	inline void VectorToTensorInPlace(const std::vector<T>& data, const std::vector<int64_t>& shape, torch::Tensor& out) {
		int64_t expected = 1;
		for (auto s : shape) expected *= s;
		
		if (expected != (int64_t)data.size() || data.empty()) {
			out = torch::tensor(data);
			return;
		}

		// Réutiliser le tensor existant s'il a la bonne taille
		if (out.defined() && out.numel() == expected && out.dtype() == torch::CppTypeToScalarType<T>()) {
			std::memcpy(out.data_ptr<T>(), data.data(), data.size() * sizeof(T));
		} else {
			out = torch::empty(shape, GetCachedOptions<T>());
			std::memcpy(out.data_ptr<T>(), data.data(), data.size() * sizeof(T));
		}
	}

	template <typename T>
	inline std::vector<T> TENSOR_TO_VEC(torch::Tensor tensor) {
		assert(tensor.dim() == 1);
		tensor = tensor.contiguous().cpu().detach().to(torch::CppTypeToScalarType<T>());
		T* data = tensor.data_ptr<T>();
		return std::vector<T>(data, data + tensor.size(0));
	}
	
	// OPTIMISATION: Version qui copie dans un vecteur existant pour éviter les allocations
	template <typename T>
	inline void TENSOR_TO_VEC_INPLACE(torch::Tensor tensor, std::vector<T>& out) {
		assert(tensor.dim() == 1);
		tensor = tensor.contiguous().cpu().detach().to(torch::CppTypeToScalarType<T>());
		T* data = tensor.data_ptr<T>();
		size_t size = tensor.size(0);
		out.resize(size);
		std::memcpy(out.data(), data, sizeof(T) * size);
	}
	
	// NOUVELLE FONCTIONNALITÉ MAJEURE: CUDA Stream Manager simplifié
#ifdef RG_CUDA_SUPPORT
	class CUDAStreamManager {
	public:
		c10::cuda::CUDAStream transferStream;
		bool initialized = false;
		
		CUDAStreamManager() : 
			transferStream(c10::cuda::getStreamFromPool()) {
			initialized = true;
		}
		
		// Exécute une fonction sur le stream de transfert
		template<typename Func>
		void RunOnTransferStream(Func&& func) {
			if (!initialized) {
				func();
				return;
			}
			
			c10::cuda::CUDAStreamGuard guard(transferStream);
			func();
		}
		
		// Attend que tous les transferts soient terminés
		void WaitTransfers() {
			if (!initialized) return;
			transferStream.synchronize();
		}
	};
#else
	// Fallback pour quand CUDA n'est pas disponible
	class CUDAStreamManager {
	public:
		bool initialized = false;
		
		template<typename Func>
		void RunOnTransferStream(Func&& func) { func(); }
		
		void WaitTransfers() {}
	};
#endif

	// Instance globale du stream manager
	inline CUDAStreamManager& GetStreamManager() {
		static CUDAStreamManager instance;
		return instance;
	}
	
	// NOUVELLE FONCTIONNALITÉ: Batch transfer helper pour transferts multiples efficaces
	struct BatchTransfer {
		std::vector<torch::Tensor*> tensors;
		torch::Device targetDevice;
		
		BatchTransfer(torch::Device dev) : targetDevice(dev) {}
		
		void Add(torch::Tensor& t) {
			tensors.push_back(&t);
		}
		
		// Transfert tous les tenseurs vers le device cible avec non_blocking
		void Execute() {
			for (auto* t : tensors) {
				if (t->defined() && t->device() != targetDevice) {
					*t = t->to(targetDevice, /*non_blocking=*/true);
				}
			}
		}
		
		// Transfert asynchrone sur stream séparé
		void ExecuteAsync() {
			auto& streamMgr = GetStreamManager();
			streamMgr.RunOnTransferStream([this]() {
				Execute();
			});
		}
	};
	
	// NOUVELLE FONCTIONNALITÉ: Pool de tenseurs pré-alloués pour éviter les allocations
	template<typename T>
	class TensorPool {
	public:
		std::vector<torch::Tensor> pool;
		std::vector<bool> inUse;
		std::vector<int64_t> defaultShape;
		torch::Device device;
		
		TensorPool(torch::Device dev, const std::vector<int64_t>& shape, size_t initialSize = 4) 
			: device(dev), defaultShape(shape) {
			pool.reserve(initialSize);
			inUse.reserve(initialSize);
			
			for (size_t i = 0; i < initialSize; i++) {
				auto opts = torch::TensorOptions()
					.dtype(torch::CppTypeToScalarType<T>())
					.device(dev);
				pool.push_back(torch::empty(shape, opts));
				inUse.push_back(false);
			}
		}
		
		// Acquiert un tensor du pool
		torch::Tensor Acquire() {
			for (size_t i = 0; i < pool.size(); i++) {
				if (!inUse[i]) {
					inUse[i] = true;
					return pool[i];
				}
			}
			
			// Pool plein, allouer un nouveau
			auto opts = torch::TensorOptions()
				.dtype(torch::CppTypeToScalarType<T>())
				.device(device);
			pool.push_back(torch::empty(defaultShape, opts));
			inUse.push_back(true);
			return pool.back();
		}
		
		// Libère un tensor
		void Release(const torch::Tensor& t) {
			for (size_t i = 0; i < pool.size(); i++) {
				if (pool[i].data_ptr() == t.data_ptr()) {
					inUse[i] = false;
					return;
				}
			}
		}
		
		// Libère tous les tensors
		void ReleaseAll() {
			std::fill(inUse.begin(), inUse.end(), false);
		}
	};
}