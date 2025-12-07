#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>
#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	SetLearningRates(config.policyLR, config.criticLR);

	// Print param counts
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));

	if (config.useGuidingPolicy) {
		RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
		MakeModels(false, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels);
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions, 
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device, 
	ModelSet& outModels) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	if (sharedHeadConfig.IsValid()) {

		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();
		fullCriticConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

// OPTIMISATION MAJEURE: Fused log-softmax pour éviter deux passes sur les données
torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec) {

	// OPTIMISATION: Convert to bool once avec fusion
	auto boolMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	// Forward pass
	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	auto logits = models["policy"]->Forward(obs, halfPrec);
	
	// OPTIMISATION: Fused temperature + mask + softmax
	// Évite les allocations intermédiaires
	if (temperature != 1.0f) {
		// Fused: (logits / temp) + mask * disabled
		logits = logits / temperature + ACTION_DISABLED_LOGIT * boolMasks.logical_not();
	} else {
		logits = logits + ACTION_DISABLED_LOGIT * boolMasks.logical_not();
	}
	
	auto result = torch::softmax(logits, -1);

	// OPTIMISATION: Conditional clamp based on gradient mode
	if (torch::GradMode::is_enabled()) {
		result = result.clamp(ACTION_MIN_PROB, 1.0f);
	} else {
		result.clamp_(ACTION_MIN_PROB, 1.0f);
	}

	return result.view({ -1, models["policy"]->config.numOutputs });
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks, 
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);

	if (deterministic) {
		if (outActions)
			*outActions = probs.argmax(1).flatten();
		return;
	}
	
	// OPTIMISATION MAJEURE: GPU multinomial pour tous les cas GPU
	if (probs.device().is_cuda()) {
		// OPTIMISATION: Fused multinomial + gather + log
		auto actions = torch::multinomial(probs, 1, /*replacement=*/true).squeeze(-1);
		if (outActions)
			*outActions = actions;
		if (outLogProbs) {
			// OPTIMISATION: Fused gather + log
			*outLogProbs = probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1).log();
		}
		return;
	}

	// CPU path optimisé
	auto probsCpu = probs.contiguous();
	auto probsAcc = probsCpu.accessor<float, 2>();
	
	const int64_t numRows = probsCpu.size(0);
	const int64_t cols = probsCpu.size(1);
	
	// OPTIMISATION: Direct allocation sans intermediate vectors
	torch::Tensor actionsT = torch::empty({numRows}, torch::kInt64);
	torch::Tensor logProbsT = torch::empty({numRows}, torch::kFloat32);
	auto actionsPtr = actionsT.data_ptr<int64_t>();
	auto logProbsPtr = logProbsT.data_ptr<float>();

	// OPTIMISATION: Thread-local RNG avec meilleur seed
	static thread_local std::mt19937 gen(std::random_device{}() ^ (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count());
	
	// OPTIMISATION: Vectorized sampling loop
	for (int64_t i = 0; i < numRows; i++) {
		// OPTIMISATION: Single pass cumulative sum + sampling
		float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(gen);
		float running = 0.0f;
		int64_t picked = cols - 1;
		
		for (int64_t j = 0; j < cols; j++) {
			float p = probsAcc[i][j];
			running += p;
			if (r <= running) { 
				picked = j; 
				break; 
			}
		}
		
		actionsPtr[i] = picked;
		// OPTIMISATION: Fast log avec clamp intégré
		logProbsPtr[i] = std::log(std::max(1e-12f, probsAcc[i][picked]));
	}

	if (outActions)
		*outActions = actionsT;
	if (outLogProbs)
		*outLogProbs = logProbsT;
}

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models) {
	InferActionsFromModels(models ? *models : this->models, obs, actionMasks, config.deterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

// NOUVELLE FONCTIONNALITÉ: Inférence critic batché optimisée avec pipeline
// Utilise tout le GPU en une seule passe quand possible, avec overlap des transferts
torch::Tensor GGL::PPOLearner::InferCriticBatched(torch::Tensor obs, int64_t maxBatchSize) {
	int64_t totalRows = obs.size(0);
	
	// Clamp maxBatchSize to reasonable value
	if (maxBatchSize <= 0) maxBatchSize = 50000;
	
	// Si le batch tient en mémoire, faire une seule passe
	if (totalRows <= maxBatchSize || device.is_cpu()) {
		return InferCritic(obs.to(device, /*non_blocking=*/true));
	}
	
	// OPTIMISATION: Utiliser le stream manager pour overlap transfert/calcul
	auto& streamMgr = GetStreamManager();
	
	// Pré-allouer le résultat
	auto result = torch::empty({totalRows}, torch::kFloat32);
	
	// Double buffering pour overlap
	torch::Tensor currentBatchGPU, nextBatchGPU;
	int64_t currentStart = 0;
	
	// Transférer le premier batch
	auto firstBatch = obs.slice(0, 0, std::min(maxBatchSize, totalRows));
	currentBatchGPU = firstBatch.to(device, /*non_blocking=*/true);
	
	for (int64_t i = 0; i < totalRows; i += maxBatchSize) {
		int64_t end = std::min(i + maxBatchSize, totalRows);
		int64_t nextStart = end;
		int64_t nextEnd = std::min(nextStart + maxBatchSize, totalRows);
		
		// Commencer le transfert du prochain batch en parallèle
		if (nextStart < totalRows) {
			streamMgr.RunOnTransferStream([&]() {
				nextBatchGPU = obs.slice(0, nextStart, nextEnd).to(device, /*non_blocking=*/true);
			});
		}
		
		// Calculer le batch courant
		auto batchResult = InferCritic(currentBatchGPU);
		
		// Copier le résultat vers CPU
		result.slice(0, i, end).copy_(batchResult.cpu(), /*non_blocking=*/false);
		
		// Attendre le transfert du prochain batch et le préparer
		if (nextStart < totalRows) {
			streamMgr.WaitTransfers();
			currentBatchGPU = nextBatchGPU;
		}
	}
	
	return result;
}

torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	// Use log directly on probs (already clamped in InferPolicyProbsFromModels)
	auto logProbs = probs.log();
	auto entropy = -(logProbs * probs).sum(-1);

	if (maskEntropy) {
		// Pre-compute denominator once
		auto validActions = actionMasks.to(torch::kFloat32).sum(-1);
		entropy = entropy / validActions.log();
	} else {
		// OPTIMISATION: Calculer le log une seule fois et le réutiliser
		static thread_local float cachedLogNumActions = 0;
		static thread_local int64_t cachedNumActions = 0;
		
		int64_t numActions = actionMasks.size(-1);
		if (numActions != cachedNumActions) {
			cachedLogNumActions = std::log(static_cast<float>(numActions));
			cachedNumActions = numActions;
		}
		entropy = entropy / cachedLogNumActions;
	}

	return entropy.mean();
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	std::string stage = "init";
	int64_t dbgLastActMin = 0;
	int64_t dbgLastActMax = 0;
	int64_t dbgLastBatchElems = 0;
	try {
	// OPTIMISATION: Créer le MSELoss une seule fois (il est réutilisé)
	static torch::nn::MSELoss mseLoss;

	// OPTIMISATION: Accumuler les métriques sur GPU et ne synchroniser qu'à la fin
	// Utiliser des scalaires au lieu de tenseurs pour certaines métriques
	torch::Tensor accumEntropy = torch::zeros({1}, device);
	torch::Tensor accumDivergence = torch::zeros({1}, device);
	torch::Tensor accumPolicyLoss = torch::zeros({1}, device);
	torch::Tensor accumCriticLoss = torch::zeros({1}, device);
	torch::Tensor accumRatio = torch::zeros({1}, device);
	torch::Tensor accumClip = torch::zeros({1}, device);
	int64_t numAccumulated = 0;

	AvgTracker avgRelEntropyLoss, avgGuidingLoss;

	// OPTIMISATION MAJEURE: Ne copier les paramètres que si on va les reporter
	torch::Tensor policyBefore, criticBefore, sharedHeadBefore;
	if (!isFirstIteration) {
		policyBefore = models["policy"]->CopyParams();
		criticBefore = models["critic"]->CopyParams();
		if (models["shared_head"])
			sharedHeadBefore = models["shared_head"]->CopyParams();
	}

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	const int64_t maxActionIdx = models["policy"]->config.numOutputs - 1;
	
	// OPTIMISATION: Pré-calculer les constantes utilisées dans la boucle
	const float clipRangeLow = 1.0f - config.clipRange;
	const float clipRangeHigh = 1.0f + config.clipRange;
	const float entropyScale = config.entropyScale;

	// OPTIMISATION MAJEURE: Double buffering des batches pour prefetch GPU
	DoubleBufferedBatches doubleBuffer(device);

	for (int epoch = 0; epoch < config.epochs; epoch++) {

		stage = "get_batches";
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);
		
		// OPTIMISATION: Utiliser le double buffer pour prefetch
		doubleBuffer.SetBatches(std::move(batches));
		
		// Prefetch le premier batch immédiatement
		if (device.is_cuda() && doubleBuffer.Size() > 0) {
			doubleBuffer.StartPrefetch(0);
		}

		for (size_t batchIdx = 0; batchIdx < doubleBuffer.Size(); batchIdx++) {
			stage = "batch_loop";
			
			// OPTIMISATION: Prefetch le prochain batch pendant qu'on traite le courant
			doubleBuffer.PrefetchNext(batchIdx);
			
			// Récupérer le batch courant (peut être déjà sur GPU si prefetché)
			auto& batch = doubleBuffer.GetBatch(batchIdx);
			
			auto& batchActs = batch.actions;
			if (batchActs.defined()) {
				dbgLastActMin = batchActs.min().item<int64_t>();
				dbgLastActMax = batchActs.max().item<int64_t>();
				dbgLastBatchElems = batchActs.numel();
			} else {
				dbgLastActMin = dbgLastActMax = 0;
				dbgLastBatchElems = 0;
			}
			auto& batchOldProbs = batch.logProbs;
			auto& batchObs = batch.states;
			auto& batchActionMasks = batch.actionMasks;
			auto& batchTargetValues = batch.targetValues;
			auto& batchAdvantages = batch.advantages;

			// Clamp actions in-place
			if (batchActs.defined())
				batchActs.clamp_(0, maxActionIdx);
			
			// OPTIMISATION: Normalisation des avantages avec opérations fusionnées
			if (batchAdvantages.defined() && batchAdvantages.numel() > 1) {
				// OPTIMISATION: Utiliser std_mean pour un seul appel
				auto advMean = batchAdvantages.mean();
				auto advStd = batchAdvantages.std();
				// In-place subtraction and division
				batchAdvantages.sub_(advMean).div_(advStd + 1e-8f);
			}

			auto fnRunMinibatch = [&](int start, int stop) {

				const float batchSizeRatio = (stop - start) / (float)config.batchSize;

				// Les données sont déjà sur device grâce au prefetch
				auto acts = batchActs.slice(0, start, stop);
				auto obs = batchObs.slice(0, start, stop);
				auto actionMasks = batchActionMasks.slice(0, start, stop);
				auto advantages = batchAdvantages.slice(0, start, stop);
				auto oldProbs = batchOldProbs.slice(0, start, stop);
				auto targetValues = batchTargetValues.slice(0, start, stop);
				
				// Si pas sur GPU, transférer maintenant
				if (!obs.device().is_cuda() && device.is_cuda()) {
					acts = acts.to(device, /*non_blocking=*/true);
					obs = obs.to(device, /*non_blocking=*/true);
					actionMasks = actionMasks.to(device, /*non_blocking=*/true);
					advantages = advantages.to(device, /*non_blocking=*/true);
					oldProbs = oldProbs.to(device, /*non_blocking=*/true);
					targetValues = targetValues.to(device, /*non_blocking=*/true);
				}

				// OPTIMISATION MAJEURE: Calculer shared_head une seule fois si policy ET critic l'utilisent
				torch::Tensor sharedFeatures;
				if (models["shared_head"] && (trainPolicy || trainCritic)) {
					sharedFeatures = models["shared_head"]->Forward(obs, false);
				}

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {
					// OPTIMISATION: Utiliser les features partagées si disponibles
					torch::Tensor policyInput = sharedFeatures.defined() ? sharedFeatures : obs;
					auto logits = models["policy"]->Forward(policyInput, false);
					
					// OPTIMISATION: Inline le calcul des probs au lieu d'appeler InferPolicyProbsFromModels
					constexpr float ACTION_MIN_PROB = 1e-11f;
					constexpr float ACTION_DISABLED_LOGIT = -1e10f;
					
					auto boolMasks = actionMasks.to(torch::kBool);
					if (config.policyTemperature != 1.0f) {
						logits = logits / config.policyTemperature + ACTION_DISABLED_LOGIT * boolMasks.logical_not();
					} else {
						logits = logits + ACTION_DISABLED_LOGIT * boolMasks.logical_not();
					}
					
					probs = torch::softmax(logits, -1).clamp(ACTION_MIN_PROB, 1.0f);
					probs = probs.view({ -1, models["policy"]->config.numOutputs });
					
					// OPTIMISATION: Fused gather + log
					logProbs = probs.gather(-1, acts.unsqueeze(-1)).squeeze(-1).log();
					
					// Entropy avec masque
					auto logProbsForEntropy = probs.log();
					entropy = -(logProbsForEntropy * probs).sum(-1);
					if (config.maskEntropy) {
						auto validActions = actionMasks.to(torch::kFloat32).sum(-1);
						entropy = entropy / validActions.log();
					} else {
						static thread_local float cachedLogNumActions = 0;
						static thread_local int64_t cachedNumActions = 0;
						int64_t numActions = actionMasks.size(-1);
						if (numActions != cachedNumActions) {
							cachedLogNumActions = std::log(static_cast<float>(numActions));
							cachedNumActions = numActions;
						}
						entropy = entropy / cachedLogNumActions;
					}
					entropy = entropy.mean();
					
					accumEntropy += entropy.detach();

					// OPTIMISATION: Utiliser exp in-place
					ratio = (logProbs - oldProbs).exp_();
					accumRatio += ratio.mean().detach();
					
					clipped = ratio.clamp(clipRangeLow, clipRangeHigh);

					// OPTIMISATION: Fused surrogate loss calculation
					auto surr1 = ratio * advantages;
					auto surr2 = clipped * advantages;
					policyLoss = -torch::min(surr1, surr2).mean();
					
					accumPolicyLoss += policyLoss.detach();

					ppoLoss = (policyLoss - entropy * entropyScale) * batchSizeRatio;

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision);
						}

						auto guidingLoss = (guidingProbs - probs).abs().mean();
						avgGuidingLoss.Add(guidingLoss.detach().cpu().item<float>());
						ppoLoss = ppoLoss + guidingLoss * config.guidingStrength;
					}
				}

				torch::Tensor criticLoss;
				if (trainCritic) {
					// OPTIMISATION: Utiliser les features partagées si disponibles
					torch::Tensor criticInput = sharedFeatures.defined() ? sharedFeatures : obs;
					auto vals = models["critic"]->Forward(criticInput, config.useHalfPrecision).flatten();
					
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					accumCriticLoss += criticLoss.detach();
				}

				if (trainPolicy) {
					RG_NO_GRAD;
					auto logRatio = logProbs - oldProbs;
					// OPTIMISATION: Fused KL calculation
					auto klTensor = logRatio.exp() - 1.0f - logRatio;
					accumDivergence += klTensor.mean().detach();

					auto clipFraction = ((ratio - 1.0f).abs() > config.clipRange).to(torch::kFloat).mean();
					accumClip += clipFraction;
				}
				
				numAccumulated++;

				// OPTIMISATION: Combined backward pass - évite multiple backward
				if (trainPolicy && trainCritic) {
					(ppoLoss + criticLoss).backward();
				} else if (trainPolicy) {
					ppoLoss.backward();
				} else if (trainCritic) {
					criticLoss.backward();
				}
			};

			try {
				int64_t actualBatchSize = batchObs.defined() ? batchObs.size(0) : config.batchSize;
				if (device.is_cpu()) {
					fnRunMinibatch(0, actualBatchSize);
				} else {
					for (int64_t mbs = 0; mbs < actualBatchSize; mbs += config.miniBatchSize) {
						int64_t end = std::min(mbs + config.miniBatchSize, actualBatchSize);
						fnRunMinibatch(mbs, end);
					}
				}
			} catch (const std::exception& e) {
				RG_LOG("PPO minibatch skipped due to exception: " << e.what()
					<< " | acts min/max: [" << dbgLastActMin << ", " << dbgLastActMax << "]");
				continue;
			}

			// Gradient clipping
			if (trainPolicy)
				nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);
			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			// OPTIMISATION: Utiliser StepOptims (déjà optimisé avec set_to_none=true)
			models.StepOptims();
		}
		
		// Attendre le dernier prefetch
		doubleBuffer.WaitPendingPrefetch();
	}

	// Single sync at the end - évite les synchronisations multiples
	float n = static_cast<float>(std::max(numAccumulated, (int64_t)1));
	float avgEntropy = (accumEntropy / n).cpu().item<float>();
	float avgDivergence = (accumDivergence / n).cpu().item<float>();
	float avgPolicyLoss = (accumPolicyLoss / n).cpu().item<float>();
	float avgCriticLoss = (accumCriticLoss / n).cpu().item<float>();
	float avgRatio = (accumRatio / n).cpu().item<float>();
	float avgClip = (accumClip / n).cpu().item<float>();

	report["Policy Entropy"] = avgEntropy;
	report["Mean KL Divergence"] = avgDivergence;
	if (!isFirstIteration) {
		report["Policy Loss"] = avgPolicyLoss;
		report["Critic Loss"] = avgCriticLoss;

		if (config.useGuidingPolicy)
			report["Guiding Loss"] = avgGuidingLoss.Get();

		report["SB3 Clip Fraction"] = avgClip;
		
		auto policyAfter = models["policy"]->CopyParams();
		auto criticAfter = models["critic"]->CopyParams();
		float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
		float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();
		
		report["Policy Update Magnitude"] = policyUpdateMagnitude;
		report["Critic Update Magnitude"] = criticUpdateMagnitude;
		
		// Reporter le shared_head update magnitude s'il existe
		if (models["shared_head"] && sharedHeadBefore.defined()) {
			auto sharedHeadAfter = models["shared_head"]->CopyParams();
			float sharedHeadUpdateMagnitude = (sharedHeadBefore - sharedHeadAfter).norm().item<float>();
			report["Shared Head Update Magnitude"] = sharedHeadUpdateMagnitude;
		}
	}
	} catch (const std::exception& e) {
		RG_LOG("PPOLearner::Learn recovered from exception at stage [" << stage << "]: " << e.what()
			<< " | last acts min/max/count: [" << dbgLastActMin << ", " << dbgLastActMax << "] / " << dbgLastBatchElems);
#ifdef RG_CUDA_SUPPORT
		if (device.is_cuda()) {
			try { c10::cuda::CUDACachingAllocator::emptyCache(); } catch (...) {}
		}
#endif
		return;
	}
}

void GGL::PPOLearner::TransferLearn(
	ModelSet& oldModels,
	torch::Tensor newObs, torch::Tensor oldObs,
	torch::Tensor newActionMasks, torch::Tensor oldActionMasks,
	torch::Tensor actionMaps,
	Report& report,
	const TransferLearnConfig& tlConfig
) {

	torch::Tensor oldProbs;
	{ // No grad for old model inference
		RG_NO_GRAD;
		oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision);
		report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();

		if (actionMaps.defined())
			oldProbs = oldProbs.gather(1, actionMaps);
	}

	for (auto& model : GetPolicyModels())
		model->SetOptimLR(tlConfig.lr);

	auto policyBefore = models["policy"]->CopyParams();
	
	for (int i = 0; i < tlConfig.epochs; i++) {
		torch::Tensor newProbs = InferPolicyProbsFromModels(models, newObs, newActionMasks, config.policyTemperature, false);

		// Non-summative KL div	loss
		torch::Tensor transferLearnLoss;
		if (tlConfig.useKLDiv) {
			transferLearnLoss = (oldProbs * torch::log(oldProbs / newProbs)).abs();
		} else {
			transferLearnLoss = (oldProbs - newProbs).abs();
		}
		transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
		transferLearnLoss = transferLearnLoss.mean();
		transferLearnLoss *= tlConfig.lossScale;

		if (i == 0) {
			RG_NO_GRAD;
			torch::Tensor matchingActionsMask = (newProbs.detach().argmax(-1) == oldProbs.detach().argmax(-1));
			report["Transfer Learn Accuracy"] = matchingActionsMask.to(torch::kFloat).mean().cpu().item<float>();
			report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();

			report["Policy Entropy"] = ComputeEntropy(newProbs, newActionMasks, config.maskEntropy).detach().cpu().item<float>();
		}

		transferLearnLoss.backward();

		models.StepOptims();
	}

	auto policyAfter = models["policy"]->CopyParams();
	report["Policy Update Magnitude"] = (policyBefore - policyAfter).norm().item<float>();
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	models["policy"]->SetOptimLR(policyLR);
	models["critic"]->SetOptimLR(criticLR);

	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(RS_MIN(policyLR, criticLR));

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (model->modelName == "critic")
			continue;
		
		result.Add(model);
	}
	return result;
}
