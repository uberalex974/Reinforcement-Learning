#include "GAE.h"
#include <execution>
#include <thread>
#include <future>

// OPTIMISATION MAJEURE: GAE avec calculs vectorisés, élimination des branches, et pré-calculs agressifs
void GGL::GAE::Compute(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange
) {
	const bool hasTruncValPreds = truncValPreds.defined();
	const int numReturns = static_cast<int>(rews.size(0));
	
	if (numReturns == 0) {
		outAdvantages = torch::empty({0}, torch::kFloat32);
		outReturns = torch::empty({0}, torch::kFloat32);
		outTargetValues = torch::empty({0}, torch::kFloat32);
		outRewClipPortion = 0;
		return;
	}
	
	// OPTIMISATION: Utiliser empty() au lieu de zeros()
	outAdvantages = torch::empty(numReturns, torch::kFloat32);
	outReturns = torch::empty(numReturns, torch::kFloat32);

	// Ensure contiguity once
	rews = rews.contiguous();
	terminals = terminals.contiguous();
	valPreds = valPreds.contiguous();
	if (hasTruncValPreds)
		truncValPreds = truncValPreds.contiguous();

	// Raw pointers for fast access
	const auto _terminals = terminals.const_data_ptr<int8_t>();
	const auto _rews = rews.const_data_ptr<float>();
	const auto _valPreds = valPreds.const_data_ptr<float>();
	const float* _truncValPreds = hasTruncValPreds ? truncValPreds.const_data_ptr<float>() : nullptr;
	const int numTruncs = hasTruncValPreds ? static_cast<int>(truncValPreds.size(0)) : 0;

	auto _outReturns = outReturns.data_ptr<float>();
	auto _outAdvantages = outAdvantages.data_ptr<float>();

	// Pré-calcul des constantes
	const bool shouldNormalize = (returnStd != 0 && returnStd != 1);
	const bool shouldClip = (clipRange > 0);
	const float invReturnStd = shouldNormalize ? (1.0f / returnStd) : 1.0f;
	const float gammaLambda = gamma * lambda;
	const int lastStep = numReturns - 1;

	// OPTIMISATION MAJEURE: Pré-calculer TOUTES les valeurs nécessaires en une passe
	// Évite les branches dans la boucle principale
	thread_local std::vector<float> nextValPreds;
	thread_local std::vector<float> notDoneNotTruncs;
	thread_local std::vector<float> normalizedRews;
	thread_local std::vector<int> truncStepIndices;
	
	nextValPreds.resize(numReturns);
	notDoneNotTruncs.resize(numReturns);
	
	// Construire le mapping truncation -> step
	truncStepIndices.clear();
	if (hasTruncValPreds) {
		truncStepIndices.reserve(numTruncs);
	}
	
	// PASSE 1: Construire les truncation indices
	for (int i = 0; i < numReturns; i++) {
		if (_terminals[i] == RLGC::TerminalType::TRUNCATED) {
			truncStepIndices.push_back(i);
		}
	}
	
	// PASSE 2: Pré-calculer nextValPreds et notDoneNotTruncs
	int truncIdx = 0;
	for (int step = 0; step < numReturns; step++) {
		const int8_t terminal = _terminals[step];
		
		// Calculer notDoneNotTrunc
		const float done = (terminal == RLGC::TerminalType::NORMAL) ? 1.0f : 0.0f;
		const float trunc = (terminal == RLGC::TerminalType::TRUNCATED) ? 1.0f : 0.0f;
		notDoneNotTruncs[step] = (1.0f - done) * (1.0f - trunc);
		
		// Calculer nextValPred
		if (terminal == RLGC::TerminalType::NORMAL) {
			nextValPreds[step] = 0.0f;
		} else if (terminal == RLGC::TerminalType::TRUNCATED && hasTruncValPreds) {
			// Trouver l'index dans truncValPreds
			int tidx = -1;
			for (size_t ti = 0; ti < truncStepIndices.size(); ti++) {
				if (truncStepIndices[ti] == step) {
					tidx = static_cast<int>(ti);
					break;
				}
			}
			nextValPreds[step] = (tidx >= 0 && tidx < numTruncs) ? _truncValPreds[tidx] : 0.0f;
		} else if (step < lastStep) {
			nextValPreds[step] = _valPreds[step + 1];
		} else {
			nextValPreds[step] = 0.0f;
		}
	}

	// PASSE 3 (optionnelle): Normaliser les rewards
	float localTotalRew = 0.0f;
	float localTotalClippedRew = 0.0f;
	const float* rewardsPtr;
	
	if (shouldNormalize) {
		normalizedRews.resize(numReturns);
		
		// OPTIMISATION: Loop unrolling x8 pour meilleure vectorisation
		int i = 0;
		const int unrollEnd = numReturns - (numReturns % 8);
		
		for (; i < unrollEnd; i += 8) {
			float n0 = _rews[i] * invReturnStd;
			float n1 = _rews[i+1] * invReturnStd;
			float n2 = _rews[i+2] * invReturnStd;
			float n3 = _rews[i+3] * invReturnStd;
			float n4 = _rews[i+4] * invReturnStd;
			float n5 = _rews[i+5] * invReturnStd;
			float n6 = _rews[i+6] * invReturnStd;
			float n7 = _rews[i+7] * invReturnStd;
			
			localTotalRew += std::abs(n0) + std::abs(n1) + std::abs(n2) + std::abs(n3)
			               + std::abs(n4) + std::abs(n5) + std::abs(n6) + std::abs(n7);
			
			if (shouldClip) {
				n0 = RS_CLAMP(n0, -clipRange, clipRange);
				n1 = RS_CLAMP(n1, -clipRange, clipRange);
				n2 = RS_CLAMP(n2, -clipRange, clipRange);
				n3 = RS_CLAMP(n3, -clipRange, clipRange);
				n4 = RS_CLAMP(n4, -clipRange, clipRange);
				n5 = RS_CLAMP(n5, -clipRange, clipRange);
				n6 = RS_CLAMP(n6, -clipRange, clipRange);
				n7 = RS_CLAMP(n7, -clipRange, clipRange);
			}
			
			localTotalClippedRew += std::abs(n0) + std::abs(n1) + std::abs(n2) + std::abs(n3)
			                      + std::abs(n4) + std::abs(n5) + std::abs(n6) + std::abs(n7);
			
			normalizedRews[i] = n0;
			normalizedRews[i+1] = n1;
			normalizedRews[i+2] = n2;
			normalizedRews[i+3] = n3;
			normalizedRews[i+4] = n4;
			normalizedRews[i+5] = n5;
			normalizedRews[i+6] = n6;
			normalizedRews[i+7] = n7;
		}
		
		// Remainder
		for (; i < numReturns; i++) {
			float normalized = _rews[i] * invReturnStd;
			localTotalRew += std::abs(normalized);
			if (shouldClip) {
				normalized = RS_CLAMP(normalized, -clipRange, clipRange);
			}
			localTotalClippedRew += std::abs(normalized);
			normalizedRews[i] = normalized;
		}
		
		rewardsPtr = normalizedRews.data();
	} else {
		rewardsPtr = _rews;
	}

	// PASSE 4: Boucle principale GAE (doit être séquentielle en arrière)
	// OPTIMISATION: Toutes les branches ont été éliminées grâce aux pré-calculs
	float prevLambda = 0.0f;
	float prevRet = 0.0f;

	for (int step = lastStep; step >= 0; step--) {
		const float curReward = rewardsPtr[step];
		const float nextVal = nextValPreds[step];
		const float notDoneNotTrunc = notDoneNotTruncs[step];
		const float curValPred = _valPreds[step];
		
		// Calcul du delta (sans branches)
		const float predReturn = curReward + gamma * nextVal;
		const float delta = predReturn - curValPred;
		
		// Returns (utilise raw reward, pas normalisé)
		const float curReturn = _rews[step] + prevRet * gamma * notDoneNotTrunc;
		_outReturns[step] = curReturn;
		
		// Advantage avec GAE-lambda (formule sans branches)
		prevLambda = delta + gammaLambda * notDoneNotTrunc * prevLambda;
		_outAdvantages[step] = prevLambda;
		
		prevRet = curReturn;
	}
	
	// Vérification des truncations
	if (hasTruncValPreds && truncStepIndices.size() != static_cast<size_t>(numTruncs))
		RG_ERR_CLOSE("GAE: truncation count mismatch (" << truncStepIndices.size() << "/" << numTruncs << ")");

	// OPTIMISATION: Fused addition pour target values avec torch (optimisé pour SIMD)
	outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;
	
	// Compute clip portion
	if (shouldNormalize) {
		outRewClipPortion = (localTotalRew - localTotalClippedRew) / std::max(localTotalRew, 1e-7f);
	} else {
		outRewClipPortion = 0;
	}
}

// Version GPU (fallback vers CPU pour la boucle séquentielle)
void GGL::GAE::ComputeGPU(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange,
	torch::Device device
) {
	// Pour l'instant, utiliser la version CPU puis transférer
	// (la boucle GAE est intrinsèquement séquentielle)
	Compute(rews.cpu(), terminals.cpu(), valPreds.cpu(), 
		truncValPreds.defined() ? truncValPreds.cpu() : truncValPreds,
		outAdvantages, outTargetValues, outReturns, outRewClipPortion,
		gamma, lambda, returnStd, clipRange);
	
	// Transférer les résultats sur GPU de manière asynchrone
	outAdvantages = outAdvantages.to(device, /*non_blocking=*/true);
	outTargetValues = outTargetValues.to(device, /*non_blocking=*/true);
	outReturns = outReturns.to(device, /*non_blocking=*/true);
}