#pragma once
#include "../Framework.h"
#include <atomic>

namespace GGL {
	struct AvgTracker {
		float total;
		uint64_t count;

		AvgTracker() {
			Reset();
		}

		// Returns 0 if no count
		float Get() const {
			if (count > 0) {
				return total / count;
			} else {
				return 0;
			}
		}

		void Add(float val) {
			if (!isnan(val)) {
				total += val;
				count++;
			}
		}

		AvgTracker& operator+=(float val) {
			Add(val);
			return *this;
		}

		void Add(float totalVal, uint64_t count) {
			if (!isnan(totalVal)) {
				total += totalVal;
				this->count += count;
			}
		}

		AvgTracker& operator+=(const AvgTracker& other) {
			Add(other.total, other.count);
			return *this;
		}

		void Reset() {
			total = 0;
			count = 0;
		}
	};

	// Thread-safe variant using atomics instead of mutex for better performance
	struct MutAvgTracker {
		std::atomic<double> total{0.0};
		std::atomic<uint64_t> count{0};

		MutAvgTracker() = default;

		// Returns 0 if no count
		float Get() const {
			uint64_t c = count.load(std::memory_order_relaxed);
			if (c > 0) {
				return static_cast<float>(total.load(std::memory_order_relaxed) / c);
			}
			return 0.f;
		}

		void Add(float val) {
			if (!isnan(val)) {
				// Atomic add using compare-exchange loop
				double expected = total.load(std::memory_order_relaxed);
				while (!total.compare_exchange_weak(expected, expected + val, 
					std::memory_order_relaxed, std::memory_order_relaxed)) {}
				count.fetch_add(1, std::memory_order_relaxed);
			}
		}

		MutAvgTracker& operator+=(float val) {
			Add(val);
			return *this;
		}

		void Add(float totalVal, uint64_t addCount) {
			if (!isnan(totalVal)) {
				double expected = total.load(std::memory_order_relaxed);
				while (!total.compare_exchange_weak(expected, expected + totalVal,
					std::memory_order_relaxed, std::memory_order_relaxed)) {}
				count.fetch_add(addCount, std::memory_order_relaxed);
			}
		}

		MutAvgTracker& operator+=(const MutAvgTracker& other) {
			Add(static_cast<float>(other.total.load(std::memory_order_relaxed)), 
				other.count.load(std::memory_order_relaxed));
			return *this;
		}

		void Reset() {
			total.store(0.0, std::memory_order_relaxed);
			count.store(0, std::memory_order_relaxed);
		}
	};
}