#pragma once
#include "Framework.h"

#include <thread_pool.h>

namespace RLGC {
	// Modified version of https://stackoverflow.com/questions/26516683/reusing-thread-in-loop-c
	struct ThreadPool {

		dp::thread_pool<>* _tp;
		int _numThreads;

		ThreadPool() {
			_tp = new dp::thread_pool();
			_numThreads = _tp->size();
		}

		RG_NO_COPY(ThreadPool);

		~ThreadPool() {
			delete _tp;
		}

		template <typename Function, typename... Args> requires std::invocable<Function, Args...>
		void StartJobAsync(Function&& func, Args &&...args) {
			_tp->enqueue_detach(func, args...);
		}

		void StartBatchedJobs(std::function<void(int)> func, int num, bool async) {

			for (int i = 0; i < num; i++)
				StartJobAsync(func, i);

			if (!async)
				WaitUntilDone();
		}
		
		// OPTIMISATION MAJEURE: Batched jobs avec chunks pour réduire l'overhead
		// Au lieu de créer N jobs, on crée numThreads jobs qui traitent N/numThreads éléments chacun
		void StartBatchedJobsChunked(std::function<void(int)> func, int num, bool async) {
			if (num <= 0) return;
			
			// Si peu d'éléments, utiliser la méthode standard
			if (num <= _numThreads * 2) {
				StartBatchedJobs(func, num, async);
				return;
			}
			
			// Calculer la taille des chunks
			int chunkSize = (num + _numThreads - 1) / _numThreads;
			
			for (int t = 0; t < _numThreads; t++) {
				int start = t * chunkSize;
				int end = std::min(start + chunkSize, num);
				
				if (start >= num) break;
				
				StartJobAsync([func, start, end]() {
					for (int i = start; i < end; i++) {
						func(i);
					}
				});
			}

			if (!async)
				WaitUntilDone();
		}
		
		// NOUVELLE FONCTIONNALITÉ: Parallel for avec range
		template<typename Func>
		void ParallelFor(int start, int end, Func&& func, bool async = false) {
			int num = end - start;
			if (num <= 0) return;
			
			if (num <= _numThreads * 2) {
				for (int i = start; i < end; i++) {
					StartJobAsync([&func, i]() { func(i); });
				}
			} else {
				int chunkSize = (num + _numThreads - 1) / _numThreads;
				
				for (int t = 0; t < _numThreads; t++) {
					int chunkStart = start + t * chunkSize;
					int chunkEnd = std::min(chunkStart + chunkSize, end);
					
					if (chunkStart >= end) break;
					
					StartJobAsync([&func, chunkStart, chunkEnd]() {
						for (int i = chunkStart; i < chunkEnd; i++) {
							func(i);
						}
					});
				}
			}

			if (!async)
				WaitUntilDone();
		}

		void WaitUntilDone() {
			_tp->wait_for_tasks();
		}

		int GetNumThreads() const {
			return _numThreads;
		}
	};

	extern ThreadPool g_ThreadPool;
}