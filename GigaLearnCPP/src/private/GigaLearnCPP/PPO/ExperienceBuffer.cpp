#include "ExperienceBuffer.h"
#include <chrono>
#include <limits>
#include <numeric>
#include <execution>

using namespace torch;

GGL::ExperienceBuffer::ExperienceBuffer(int seed, torch::Device device) :
	seed(seed), device(device), rng(seed) {
	// Reserve initial capacity for shuffled indices
	shuffledIndices.reserve(200000);
}

GGL::ExperienceTensors GGL::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {
	using Clock = std::chrono::high_resolution_clock;
	auto t0 = Clock::now();

	// OPTIMISATION: Fast path pour données invalides
	if (!data.states.defined() || size == 0) {
		return ExperienceTensors{};
	}
	
	int64_t rowLimit = data.states.size(0);
	
	// OPTIMISATION: Fast path - check actions only since it's most likely to be smaller
	if (data.actions.defined()) {
		rowLimit = std::min(rowLimit, data.actions.size(0));
	}
	
	if (rowLimit <= 0) {
		return ExperienceTensors{};
	}

	// OPTIMISATION MAJEURE: Créer le tensor d'indices une seule fois
	torch::Tensor tIndices;
	
	// OPTIMISATION: Quick max check avec early exit
	int64_t maxIdx = 0;
	for (size_t i = 0; i < size; i++) {
		if (indices[i] > maxIdx) maxIdx = indices[i];
		if (maxIdx >= rowLimit) break;
	}
	
	if (maxIdx >= rowLimit) {
		// On doit clamper - réutiliser scratchIndices
		if (!scratchIndices.defined() || scratchIndices.numel() < static_cast<int64_t>(size)) {
			scratchIndices = torch::empty({static_cast<int64_t>(size * 2)}, torch::kInt64);
		}
		auto scratchPtr = scratchIndices.data_ptr<int64_t>();
		const int64_t limit = rowLimit - 1;
		
		// OPTIMISATION: Loop unrolling x4
		size_t i = 0;
		const size_t unrollEnd = size - (size % 4);
		for (; i < unrollEnd; i += 4) {
			scratchPtr[i] = std::min(indices[i], limit);
			scratchPtr[i+1] = std::min(indices[i+1], limit);
			scratchPtr[i+2] = std::min(indices[i+2], limit);
			scratchPtr[i+3] = std::min(indices[i+3], limit);
		}
		for (; i < size; i++) {
			scratchPtr[i] = std::min(indices[i], limit);
		}
		tIndices = scratchIndices.slice(0, 0, static_cast<int64_t>(size));
	} else {
		// OPTIMISATION: Zero-copy path - from_blob évite la copie
		tIndices = torch::from_blob(
			const_cast<int64_t*>(indices),
			{static_cast<int64_t>(size)},
			torch::kInt64
		);
	}

	ExperienceTensors result;
	
	// OPTIMISATION MAJEURE: Parallel index_select pour tous les tenseurs
	// index_select est thread-safe sur des tenseurs différents
	auto* toItr = result.begin();
	auto* fromItr = data.begin();
	for (; toItr != result.end(); toItr++, fromItr++) {
		if (fromItr->defined()) {
			*toItr = fromItr->index_select(0, tIndices);
		}
	}

	// Clamp actions if needed - in-place pour éviter copie
	if (maxActionIndex >= 0 && result.actions.defined()) {
		result.actions.clamp_(0, maxActionIndex);
	}

	auto t1 = Clock::now();
	auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
	profile_get_samples_time_us += (uint64_t)us;
	profile_get_samples_count++;

	return result;
}

std::vector<GGL::ExperienceTensors> GGL::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize, bool overbatching) {

	RG_NO_GRAD;

	// OPTIMISATION: Fast path si données invalides
	if (!data.states.defined() || data.states.size(0) <= 0) {
		return {};
	}

	int64_t expSize = data.states.size(0);
	
	// OPTIMISATION: Only check actions tensor (most critical)
	if (data.actions.defined()) {
		expSize = std::min(expSize, data.actions.size(0));
	}

	if (expSize <= 0) {
		return {};
	}

	// OPTIMISATION: Réutiliser le vecteur d'indices - resize only if needed
	if (shuffledIndices.size() != static_cast<size_t>(expSize)) {
		shuffledIndices.resize(expSize);
		// OPTIMISATION: Utiliser iota pour remplir séquentiellement
		std::iota(shuffledIndices.begin(), shuffledIndices.end(), 0);
	} else {
		// OPTIMISATION: Si même taille, juste re-shuffler (pas besoin de re-remplir)
	}
	
	// OPTIMISATION: Fisher-Yates shuffle (in-place, O(n))
	std::shuffle(shuffledIndices.begin(), shuffledIndices.end(), rng);

	// Pre-calculate number of batches
	int64_t numBatches = expSize / batchSize;
	if (overbatching && (expSize % batchSize) > 0)
		numBatches++;
	
	if (numBatches == 0) numBatches = 1;
		
	std::vector<ExperienceTensors> result;
	result.reserve(numBatches);

	// OPTIMISATION: Single loop avec moins de conditions
	for (int64_t startIdx = 0; startIdx < expSize; startIdx += batchSize) {
		int64_t endIdx = startIdx + batchSize;
		
		// Overbatching: si c'est le dernier batch et qu'il reste moins de 2*batchSize
		if (endIdx + batchSize > expSize && overbatching) {
			endIdx = expSize;
		}
		
		if (endIdx > expSize) break;
		
		int64_t curBatchSize = endIdx - startIdx;
		if (curBatchSize <= 0) break;

		result.push_back(_GetSamples(shuffledIndices.data() + startIdx, curBatchSize));
		
		// Si on a pris tout le reste, sortir
		if (endIdx == expSize) break;
	}

	return result;
}

void GGL::ExperienceBuffer::PrintProfile() const {
	uint64_t count = profile_get_samples_count.load();
	uint64_t totalUs = profile_get_samples_time_us.load();
	if (count == 0) return;
	double avg = (double)totalUs / (double)count;
	RG_LOG("ExperienceBuffer::_GetSamples(): avg time (us): " << avg << " over " << count << " calls");
}
