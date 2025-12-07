#pragma once
#include "../FrameworkTorch.h"
#include <atomic>
#include <future>

namespace GGL {

	struct ExperienceTensors {
		torch::Tensor
			states, actions, logProbs, targetValues, actionMasks, advantages;

		auto begin() { return &states; }
		auto end() { return &advantages + 1; }
		auto begin() const { return &states; }
		auto end() const { return &advantages + 1; }
		
		// NOUVELLE FONCTIONNALITÉ: Transfert asynchrone vers un device
		ExperienceTensors ToDevice(torch::Device device, bool nonBlocking = true) const {
			ExperienceTensors result;
			auto* toItr = result.begin();
			auto* fromItr = begin();
			for (; toItr != result.end(); toItr++, fromItr++) {
				if (fromItr->defined()) {
					*toItr = fromItr->to(device, nonBlocking);
				}
			}
			return result;
		}
		
		// NOUVELLE FONCTIONNALITÉ: Vérifier si tous les tenseurs sont sur le même device
		bool IsOnDevice(torch::Device device) const {
			for (auto* itr = begin(); itr != end(); itr++) {
				if (itr->defined() && itr->device() != device) {
					return false;
				}
			}
			return true;
		}
	};

	// NOUVELLE FONCTIONNALITÉ: Double buffer pour les batches
	// Permet de préparer le prochain batch pendant le traitement du courant
	class DoubleBufferedBatches {
	public:
		std::vector<ExperienceTensors> cpuBatches;
		std::vector<ExperienceTensors> gpuBatches;
		torch::Device targetDevice;
		size_t currentIdx = 0;
		size_t prefetchedIdx = SIZE_MAX;
		std::future<void> prefetchFuture;
		bool hasPrefetch = false;
		
		DoubleBufferedBatches(torch::Device dev) : targetDevice(dev) {}
		
		void SetBatches(std::vector<ExperienceTensors>&& batches) {
			cpuBatches = std::move(batches);
			gpuBatches.clear();
			gpuBatches.resize(cpuBatches.size());
			currentIdx = 0;
			prefetchedIdx = SIZE_MAX;
			hasPrefetch = false;
		}
		
		size_t Size() const { return cpuBatches.size(); }
		
		// Prefetch le batch à l'index donné vers le GPU
		void StartPrefetch(size_t idx) {
			if (!targetDevice.is_cuda() || idx >= cpuBatches.size()) return;
			if (idx == prefetchedIdx) return; // Déjà prefetché
			
			// Attendre le prefetch précédent si nécessaire
			if (hasPrefetch && prefetchFuture.valid()) {
				prefetchFuture.wait();
			}
			
			prefetchFuture = std::async(std::launch::async, [this, idx]() {
				gpuBatches[idx] = cpuBatches[idx].ToDevice(targetDevice, true);
			});
			prefetchedIdx = idx;
			hasPrefetch = true;
		}
		
		// Récupère le batch à l'index donné (attend le prefetch si nécessaire)
		ExperienceTensors& GetBatch(size_t idx) {
			if (!targetDevice.is_cuda()) {
				return cpuBatches[idx];
			}
			
			// Si c'est le batch prefetché, attendre et retourner
			if (idx == prefetchedIdx && hasPrefetch && prefetchFuture.valid()) {
				prefetchFuture.wait();
				hasPrefetch = false;
				return gpuBatches[idx];
			}
			
			// Sinon, transfert synchrone
			if (!gpuBatches[idx].states.defined()) {
				gpuBatches[idx] = cpuBatches[idx].ToDevice(targetDevice, false);
			}
			return gpuBatches[idx];
		}
		
		// Commence le prefetch du prochain batch
		void PrefetchNext(size_t currentBatchIdx) {
			if (currentBatchIdx + 1 < cpuBatches.size()) {
				StartPrefetch(currentBatchIdx + 1);
			}
		}
		
		void WaitPendingPrefetch() {
			if (hasPrefetch && prefetchFuture.valid()) {
				prefetchFuture.wait();
				hasPrefetch = false;
			}
		}
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
	class ExperienceBuffer {
	public:

		torch::Device device;
		int seed;
		int64_t maxActionIndex = -1; // optional safety clamp

		ExperienceTensors data;

		std::default_random_engine rng;

		// Scratch tensor for indices to avoid allocating each call
		mutable torch::Tensor scratchIndices;
		
		// Reusable index vector to avoid allocation per epoch
		mutable std::vector<int64_t> shuffledIndices;
		
		// NOUVELLE FONCTIONNALITÉ: Cache des batches pour réutilisation
		mutable std::vector<ExperienceTensors> cachedBatches;
		mutable int64_t cachedBatchSize = 0;
		mutable bool cacheValid = false;

		// Basic profiling counters for _GetSamples
		mutable std::atomic<uint64_t> profile_get_samples_time_us{0};
		mutable std::atomic<uint64_t> profile_get_samples_count{0};

		ExperienceBuffer(int seed, torch::Device device);

		ExperienceTensors _GetSamples(const int64_t* indices, size_t size) const;

		// Not const because it uses our random engine
		std::vector<ExperienceTensors> GetAllBatchesShuffled(int64_t batchSize, bool overbatching);
		
		// NOUVELLE FONCTIONNALITÉ: Invalider le cache quand les données changent
		void InvalidateCache() { cacheValid = false; }

		// Print basic profiling stats
		void PrintProfile() const;
	};
}