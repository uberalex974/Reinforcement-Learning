#pragma once
#include "../Framework.h"
#include <span>
#include <cstring>

namespace RLGC {
	typedef std::vector<float> FList;
	typedef std::vector<int> IList;
}

// Vector append operator
template <typename T>
inline std::vector<T>& operator +=(std::vector<T>& vecA, const std::vector<T>& vecB) {
	vecA.insert(vecA.end(), vecB.begin(), vecB.end());
	return vecA;
}

inline RLGC::FList& operator +=(RLGC::FList& list, float val) {
	list.push_back(val);
	return list;
}

inline RLGC::FList& operator +=(RLGC::FList& list, const Vec& val) {
	list.push_back(val.x);
	list.push_back(val.y);
	list.push_back(val.z);
	return list;
}

///////////////

namespace RLGC {
	template <typename T>
	struct DimList2 {
		size_t size[2];
		size_t numel;
		std::vector<T> data;

		DimList2() {
			memset(size, 0, sizeof(size));
			numel = 0;
		}

		DimList2(size_t size0, size_t size1) {
			size[0] = size0;
			size[1] = size1;
			numel = size[0] * size[1];
			data.resize(numel);
		}

		// OPTIMISATION: Inline pour performance
		size_t ResolveIdx(size_t idx0, size_t idx1) const {
			return idx0 * size[1] + idx1;
		}

		T& At(size_t idx0, size_t idx1) { return data[ResolveIdx(idx0, idx1)]; }
		T At(size_t idx0, size_t idx1) const { return data[ResolveIdx(idx0, idx1)]; }

		// Retourne une copie (compatibilité)
		std::vector<T> GetRow(size_t idx0) const {
			auto startItr = data.begin() + (idx0 * size[1]);
			return std::vector<T>(startItr, startItr + size[1]);
		}
		
		// OPTIMISATION MAJEURE: Retourne un span (pas de copie, C++20)
		std::span<const T> GetRowSpan(size_t idx0) const {
			return std::span<const T>(data.data() + idx0 * size[1], size[1]);
		}
		
		std::span<T> GetRowSpan(size_t idx0) {
			return std::span<T>(data.data() + idx0 * size[1], size[1]);
		}
		
		// OPTIMISATION: Retourne un pointeur brut (performance maximale)
		const T* GetRowPtr(size_t idx0) const {
			return data.data() + idx0 * size[1];
		}
		
		T* GetRowPtr(size_t idx0) {
			return data.data() + idx0 * size[1];
		}

		void Add(const std::vector<T>& newRow) {
			RG_ASSERT(size[1] == newRow.size());
			size[0]++;
			numel += size[1];
			data.insert(data.end(), newRow.begin(), newRow.end());
		}

		// OPTIMISATION MAJEURE: Set avec memcpy pour types triviaux
		void Set(size_t idx0, const std::vector<T>& newRow) {
			RG_ASSERT(size[1] == newRow.size());
			if constexpr (std::is_trivially_copyable_v<T>) {
				std::memcpy(data.data() + idx0 * size[1], newRow.data(), size[1] * sizeof(T));
			} else {
				std::copy(newRow.begin(), newRow.end(), data.begin() + idx0 * size[1]);
			}
		}
		
		// NOUVELLE FONCTIONNALITÉ: Set avec span (évite la création de vecteur temporaire)
		void Set(size_t idx0, std::span<const T> newRow) {
			RG_ASSERT(size[1] == newRow.size());
			if constexpr (std::is_trivially_copyable_v<T>) {
				std::memcpy(data.data() + idx0 * size[1], newRow.data(), size[1] * sizeof(T));
			} else {
				std::copy(newRow.begin(), newRow.end(), data.begin() + idx0 * size[1]);
			}
		}
		
		// NOUVELLE FONCTIONNALITÉ: Set avec pointeur brut
		void SetFromPtr(size_t idx0, const T* src, size_t count) {
			RG_ASSERT(count == size[1]);
			if constexpr (std::is_trivially_copyable_v<T>) {
				std::memcpy(data.data() + idx0 * size[1], src, count * sizeof(T));
			} else {
				std::copy(src, src + count, data.begin() + idx0 * size[1]);
			}
		}
		
		// NOUVELLE FONCTIONNALITÉ: Copie d'un row vers un autre dans la même liste
		void CopyRow(size_t srcIdx, size_t dstIdx) {
			if (srcIdx == dstIdx) return;
			if constexpr (std::is_trivially_copyable_v<T>) {
				std::memcpy(data.data() + dstIdx * size[1], data.data() + srcIdx * size[1], size[1] * sizeof(T));
			} else {
				std::copy_n(data.begin() + srcIdx * size[1], size[1], data.begin() + dstIdx * size[1]);
			}
		}
		
		// NOUVELLE FONCTIONNALITÉ: Remplir un row avec une valeur
		void FillRow(size_t idx0, T value) {
			std::fill_n(data.begin() + idx0 * size[1], size[1], value);
		}
		
		// NOUVELLE FONCTIONNALITÉ: Réserver de la capacité sans changer la taille
		void Reserve(size_t numRows) {
			data.reserve(numRows * size[1]);
		}
		
		// NOUVELLE FONCTIONNALITÉ: Redimensionner
		void Resize(size_t newSize0, size_t newSize1) {
			size[0] = newSize0;
			size[1] = newSize1;
			numel = size[0] * size[1];
			data.resize(numel);
		}
		
		// NOUVELLE FONCTIONNALITÉ: Clear sans désallouer
		void Clear() {
			size[0] = 0;
			numel = 0;
			data.clear();
		}

		bool Defined() const {
			return size[0] > 0 && size[1] > 0;
		}
		
		// NOUVELLE FONCTIONNALITÉ: Nombre de rows
		size_t NumRows() const {
			return size[0];
		}
		
		// NOUVELLE FONCTIONNALITÉ: Largeur de chaque row
		size_t RowWidth() const {
			return size[1];
		}
	};
}