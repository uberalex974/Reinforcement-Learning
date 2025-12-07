#include "Learner.h"

#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>

#include <torch/cuda.h>
#include <nlohmann/json.hpp>
#include <Python.h>
#include <pybind11/embed.h>

#ifdef RG_CUDA_SUPPORT
#include <c10/cuda/CUDACachingAllocator.h>
#endif
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PolicyVersionManager.h>

#include "Util/KeyPressDetector.h"
#include <private/GigaLearnCPP/Util/WelfordStat.h>
#include "Util/AvgTracker.h"

#include <future>

using namespace RLGC;

GGL::Learner::Learner(EnvCreateFn envCreateFn, LearnerConfig config, StepCallbackFn stepCallback) :
	envCreateFn(envCreateFn), config(config), stepCallback(stepCallback)
{
	if (!Py_IsInitialized()) {
		pybind11::initialize_interpreter();
		ownsInterpreter = true;
	} else {
		ownsInterpreter = false;
		RG_LOG("Python interpreter already initialized, skipping pybind11::initialize_interpreter()");
	}

#ifndef NDEBUG
	RG_LOG("===========================");
	RG_LOG("WARNING: GigaLearn runs extremely slowly in debug, and there are often bizzare issues with debug-mode torch.");
	RG_LOG("It is recommended that you compile in release mode without optimization for debugging.");
	RG_SLEEP(1000);
#endif

	if (config.tsPerSave == 0)
		config.tsPerSave = config.ppo.tsPerItr;

	RG_LOG("Learner::Learner():");

	if (config.randomSeed == -1)
		config.randomSeed = RS_CUR_MS();

	RG_LOG("\tCheckpoint Save/Load Dir: " << config.checkpointFolder);

	torch::manual_seed(config.randomSeed);

	at::Device device = at::Device(at::kCPU);
	if (
		config.deviceType == LearnerDeviceType::GPU_CUDA || 
		(config.deviceType == LearnerDeviceType::AUTO && torch::cuda::is_available())
		) {
		RG_LOG("\tUsing CUDA GPU device...");

		// Test out moving a tensor to GPU and back to make sure the device is working
		torch::Tensor t;
		bool deviceTestFailed = false;
		try {
			t = torch::tensor(0);
			t = t.to(at::Device(at::kCUDA));
			t = t.cpu();
		} catch (...) {
			deviceTestFailed = true;
		}

		if (!torch::cuda::is_available() || deviceTestFailed)
			RG_ERR_CLOSE(
				"Learner::Learner(): Can't use CUDA GPU because " <<
				(torch::cuda::is_available() ? "libtorch cannot access the GPU" : "CUDA is not available to libtorch") << ".\n" <<
				"Make sure your libtorch comes with CUDA support, and that CUDA is installed properly."
			)
		device = at::Device(at::kCUDA);
	} else {
		RG_LOG("\tUsing CPU device...");
		device = at::Device(at::kCPU);
	}

	if (RocketSim::GetStage() != RocketSimStage::INITIALIZED) {
		RG_LOG("\tInitializing RocketSim...");
		RocketSim::Init("collision_meshes", true);
	}

	{
		RG_LOG("\tCreating envs...");
		EnvSetConfig envSetConfig = {};
		envSetConfig.envCreateFn = envCreateFn;
		envSetConfig.numArenas = config.renderMode ? 1 : config.numGames;
		envSetConfig.tickSkip = config.tickSkip;
		envSetConfig.actionDelay = config.actionDelay;
		envSetConfig.saveRewards = config.addRewardsToMetrics;
		envSet = new RLGC::EnvSet(envSetConfig);
		obsSize = envSet->state.obs.size[1];
		numActions = envSet->actionParsers[0]->GetActionAmount();
	}

	{
		if (config.standardizeReturns) {
			this->returnStat = new WelfordStat();
		} else {
			this->returnStat = NULL;
		}

		if (config.standardizeObs) {
			this->obsStat = new BatchedWelfordStat(obsSize);
		} else {
			this->obsStat = NULL;
		}
	}

	try {
		RG_LOG("\tMaking PPO learner...");
		ppo = new PPOLearner(obsSize, numActions, config.ppo, device);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Failed to create PPO learner: " << e.what());
	}

	if (config.renderMode) {
		renderSender = new RenderSender(config.renderTimeScale);
	} else {
		renderSender = NULL;
	}

	if (config.skillTracker.enabled || config.trainAgainstOldVersions)
		config.savePolicyVersions = true;

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		versionMgr = new PolicyVersionManager(
			config.checkpointFolder / "policy_versions", config.maxOldVersions, config.tsPerVersion,
			config.skillTracker, envSet->config
		);
	} else {
		versionMgr = NULL;
	}

	if (!config.checkpointFolder.empty())
		Load();

	if (config.savePolicyVersions && !config.renderMode) {
		if (config.checkpointFolder.empty())
			RG_ERR_CLOSE("Cannot save/load old policy versions with no checkpoint save folder");
		auto models = ppo->GetPolicyModels();
		versionMgr->LoadVersions(models, totalTimesteps);
	}

	if (config.sendMetrics && !config.renderMode) {
		if (!runID.empty())
			RG_LOG("\tRun ID: " << runID);
		metricSender = new MetricSender(config.metricsProjectName, config.metricsGroupName, config.metricsRunName, runID);
	} else {
		metricSender = NULL;
	}

	RG_LOG(RG_DIVIDER);
}

void GGL::Learner::SaveStats(std::filesystem::path path) {
	using namespace nlohmann;

	constexpr const char* ERROR_PREFIX = "Learner::SaveStats(): ";

	std::ofstream fOut(path);
	if (!fOut.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = {};
	j["total_timesteps"] = totalTimesteps;
	j["total_iterations"] = totalIterations;

	if (config.sendMetrics)
		j["run_id"] = metricSender->curRunID;

	if (returnStat)
		j["return_stat"] = returnStat->ToJSON();
	if (obsStat)
		j["obs_stat"] = obsStat->ToJSON();

	if (versionMgr)
		versionMgr->AddRunningStatsToJSON(j);

	std::string jStr = j.dump(4);
	fOut << jStr;
}

void GGL::Learner::LoadStats(std::filesystem::path path) {
	// TODO: Repetitive code, merge repeated code into one function called from both SaveStats() and LoadStats()

	using namespace nlohmann;
	constexpr const char* ERROR_PREFIX = "Learner::LoadStats(): ";

	std::ifstream fIn(path);
	if (!fIn.good())
		RG_ERR_CLOSE(ERROR_PREFIX << "Can't open file at " << path);

	json j = json::parse(fIn);
	totalTimesteps = j["total_timesteps"];
	totalIterations = j["total_iterations"];

	if (j.contains("run_id"))
		runID = j["run_id"];

	// FIX: Vérifier si les clés existent avant de les lire
	if (returnStat && j.contains("return_stat"))
		returnStat->ReadFromJSON(j["return_stat"]);
	if (obsStat && j.contains("obs_stat"))
		obsStat->ReadFromJSON(j["obs_stat"]);

	if (versionMgr)
		versionMgr->LoadRunningStatsFromJSON(j);
}

// Different than RLGym-PPO to show that they are not compatible
constexpr const char* STATS_FILE_NAME = "RUNNING_STATS.json";

void GGL::Learner::Save() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Save(): Cannot save because config.checkpointSaveFolder is not set");

	std::filesystem::path saveFolder = config.checkpointFolder / std::to_string(totalTimesteps);
	std::filesystem::create_directories(saveFolder);

	RG_LOG("Saving to folder " << saveFolder << "...");
	SaveStats(saveFolder / STATS_FILE_NAME);
	ppo->SaveTo(saveFolder);

	// Remove old checkpoints
	if (config.checkpointsToKeep != -1) {
		std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
		while (allSavedTimesteps.size() > config.checkpointsToKeep) {
			int64_t lowestCheckpointTS = INT64_MAX;
			for (int64_t savedTimesteps : allSavedTimesteps)
				lowestCheckpointTS = RS_MIN(lowestCheckpointTS, savedTimesteps);

			std::filesystem::path removePath = config.checkpointFolder / std::to_string(lowestCheckpointTS);
			try {
				std::filesystem::remove_all(removePath);
			} catch (std::exception& e) {
				RG_ERR_CLOSE("Failed to remove old checkpoint from " << removePath << ", exception: " << e.what());
			}
			allSavedTimesteps.erase(lowestCheckpointTS);
		}
	}

	if (versionMgr)
		versionMgr->SaveVersions();

	RG_LOG(" > Done.");
}

void GGL::Learner::Load() {
	if (config.checkpointFolder.empty())
		RG_ERR_CLOSE("Learner::Load(): Cannot load because config.checkpointLoadFolder is not set");

	RG_LOG("Loading most recent checkpoint in " << config.checkpointFolder << "...");

	int64_t highest = -1;
	std::set<int64_t> allSavedTimesteps = Utils::FindNumberedDirs(config.checkpointFolder);
	for (int64_t timesteps : allSavedTimesteps)
		highest = RS_MAX(timesteps, highest);

	if (highest != -1) {
		std::filesystem::path loadFolder = config.checkpointFolder / std::to_string(highest);
		RG_LOG(" > Loading checkpoint " << loadFolder << "...");
		LoadStats(loadFolder / STATS_FILE_NAME);
		ppo->LoadFrom(loadFolder);
		RG_LOG(" > Done.");
	} else {
		RG_LOG(" > No checkpoints found, starting new model.")
	}
}

void GGL::Learner::StartQuitKeyThread(bool& quitPressed, std::thread& outThread) {
	quitPressed = false;

	RG_LOG("Press 'Q' to save and quit!");
	outThread = std::thread(
		[&] {
			while (true) {
				char c = toupper(KeyPressDetector::GetPressedChar());
				if (c == 'Q') {
					RG_LOG("Save queued, will save and exit next iteration.");
					quitPressed = true;
				}
			}
		}
	);

	outThread.detach();
}
void GGL::Learner::StartTransferLearn(const TransferLearnConfig& tlConfig) {

	RG_LOG("Starting transfer learning...");

	// TODO: Lots of manual obs builder stuff going on which is quite volatile
	//	Although I can't really think another way to do this

	std::vector<ObsBuilder*> oldObsBuilders = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders.push_back(tlConfig.makeOldObsFn());

	// Reset all obs builders initially
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldObsBuilders[i]->Reset(envSet->state.gameStates[0]);

	std::vector<ActionParser*> oldActionParsers = {};
	for (int i = 0; i < envSet->arenas.size(); i++)
		oldActionParsers.push_back(tlConfig.makeOldActFn());

	int oldNumActions = oldActionParsers[0]->GetActionAmount();

	if (oldNumActions != numActions) {
		if (!tlConfig.mapActsFn) {
			RG_ERR_CLOSE(
				"StartTransferLearn: Old and new action parsers have a different number of actions, but tlConfig.mapActsFn is NULL.\n" <<
				"You must implement this function to translate the action indices."
			);
		};
	}

	// Determine old obs size
	int oldObsSize;
	{
		GameState testState = envSet->state.gameStates[0];
		oldObsSize = oldObsBuilders[0]->BuildObs(testState.players[0], testState).size();
	}

	ModelSet oldModels = {};
	{
		RG_NO_GRAD;
		PPOLearner::MakeModels(false, oldObsSize, oldNumActions, tlConfig.oldSharedHeadConfig, tlConfig.oldPolicyConfig, {}, ppo->device, oldModels);

		oldModels.Load(tlConfig.oldModelsPath, false, false);
	}

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

	while (true) {
		try {
			Report report = {};

			// Collect obs
			std::vector<float> allNewObs = {};
			std::vector<float> allOldObs = {};
			std::vector<uint8_t> allNewActionMasks = {};
			std::vector<uint8_t> allOldActionMasks = {};
			std::vector<int> allActionMaps = {};
			int stepsCollected;
			{
				RG_NO_GRAD;
				for (stepsCollected = 0; stepsCollected < tlConfig.batchSize; stepsCollected += envSet->state.numPlayers) {
					
					auto terminals = envSet->state.terminals; // Backup
					envSet->Reset();
					for (int i = 0; i < envSet->arenas.size(); i++) // Manually reset old obs builders
						if (terminals[i])
							oldObsBuilders[i]->Reset(envSet->state.gameStates[i]);

					torch::Tensor tActions, tLogProbs;
					torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
					torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

					envSet->StepFirstHalf(true);

					allNewObs += envSet->state.obs.data;
					allNewActionMasks += envSet->state.actionMasks.data;

					// Run all old obs and old action parser on each player
					// TODO: Could be multithreaded
					for (int arenaIdx = 0; arenaIdx < envSet->arenas.size(); arenaIdx++) {
						auto& gs = envSet->state.gameStates[arenaIdx];
						for (auto& player : gs.players) {
							allOldObs += oldObsBuilders[arenaIdx]->BuildObs(player, gs);
							allOldActionMasks += oldActionParsers[arenaIdx]->GetActionMask(player, gs);

							if (tlConfig.mapActsFn) {
								auto curMap = tlConfig.mapActsFn(player, gs);
								if (curMap.size() != numActions)
									RG_ERR_CLOSE("StartTransferLearn: Your action map must have the same size as the new action parser's actions");
								allActionMaps += curMap;
							}
						}
					}

					ppo->InferActions(
						tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), 
						&tActions, &tLogProbs
					);

					auto curActions = TENSOR_TO_VEC<int>(tActions);

					envSet->Sync();
					envSet->StepSecondHalf(curActions, false);

					if (stepCallback)
						stepCallback(this, envSet->state.gameStates, report);
				}
			}

			uint64_t prevTimesteps = totalTimesteps;
			totalTimesteps += stepsCollected;
			report["Total Timesteps"] = totalTimesteps;
			report["Collected Timesteps"] = stepsCollected;
			totalIterations++;
			report["Total Iterations"] = totalIterations;

			// Make tensors
			torch::Tensor tNewObs = GGL::VectorToTensor<float>(allNewObs, { (int64_t)allNewObs.size() / obsSize, (int64_t)obsSize }).to(ppo->device, /*non_blocking=*/true);
			torch::Tensor tOldObs = GGL::VectorToTensor<float>(allOldObs, { (int64_t)allOldObs.size() / oldObsSize, (int64_t)oldObsSize }).to(ppo->device, /*non_blocking=*/true);
			torch::Tensor tNewActionMasks = GGL::VectorToTensor<uint8_t>(allNewActionMasks, { (int64_t)allNewActionMasks.size() / numActions, (int64_t)numActions }).to(ppo->device, /*non_blocking=*/true);
			torch::Tensor tOldActionMasks = GGL::VectorToTensor<uint8_t>(allOldActionMasks, { (int64_t)allOldActionMasks.size() / oldNumActions, (int64_t)oldNumActions }).to(ppo->device, /*non_blocking=*/true);

			torch::Tensor tActionMaps = {};
			if (!allActionMaps.empty())
				tActionMaps = GGL::VectorToTensor<int>(allActionMaps, { (int64_t)allActionMaps.size() / numActions, (int64_t)numActions }).to(ppo->device, /*non_blocking=*/true);

			// Transfer learn
			ppo->TransferLearn(oldModels, tNewObs, tOldObs, tNewActionMasks, tOldActionMasks, tActionMaps, report, tlConfig);

			if (versionMgr)
				versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

			if (saveQueued) {
				if (!config.checkpointFolder.empty())
					Save();
				exit(0);
			}

			if (!config.checkpointFolder.empty()) {
				if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
					// Auto-save
					Save();
				}
			}

			report.Finish();

			if (metricSender)
				metricSender->Send(report);

			report.Display(
				{
					"Transfer Learn Accuracy",
					"Transfer Learn Loss",
					"",
					"Policy Entropy",
					"Old Policy Entropy",
					"Policy Update Magnitude",
					"",
					"Collected Timesteps",
					"Total Timesteps",
					"Total Iterations"
				}
			);
		} catch (const std::exception& e) {
			RG_LOG("Warning: recovered from transfer learn iteration exception: " << e.what());
#ifdef RG_CUDA_SUPPORT
			if (ppo && ppo->device.is_cuda()) {
				try { c10::cuda::CUDACachingAllocator::emptyCache(); } catch (...) {}
			}
#endif
			continue;
		}
	}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during transfer learn loop: " << e.what());
	}
}

void GGL::Learner::Start() {

	bool render = config.renderMode;

	RG_LOG("Learner::Start():");
	RG_LOG("\tObs size: " << obsSize);
	RG_LOG("\tAction amount: " << numActions);

	if (render)
		RG_LOG("\t(Render mode enabled)");

	try {
		bool saveQueued;
		std::thread keyPressThread;
		StartQuitKeyThread(saveQueued, keyPressThread);

		ExperienceBuffer experience = ExperienceBuffer(config.randomSeed, torch::kCPU);
		experience.maxActionIndex = numActions - 1;

		int numPlayers = envSet->state.numPlayers;

		// OPTIMISATION: Improved Trajectory struct with pre-allocated capacity and efficient append
		struct Trajectory {
			FList states, nextStates, rewards, logProbs;
			std::vector<uint8_t> actionMasks;
			std::vector<int8_t> terminals;
			std::vector<int32_t> actions;

			Trajectory() {
				Reserve(2048);
			}

			void Reserve(size_t capacity) {
				states.reserve(capacity);
				nextStates.reserve(64);
				rewards.reserve(capacity);
				logProbs.reserve(capacity);
				actionMasks.reserve(capacity);
				terminals.reserve(capacity);
				actions.reserve(capacity);
			}

			void Clear() {
				states.clear();
				nextStates.clear();
				rewards.clear();
				logProbs.clear();
				actionMasks.clear();
				terminals.clear();
				actions.clear();
			}

			void Append(const Trajectory& other) {
				states.insert(states.end(), other.states.begin(), other.states.end());
				nextStates.insert(nextStates.end(), other.nextStates.begin(), other.nextStates.end());
				rewards.insert(rewards.end(), other.rewards.begin(), other.rewards.end());
				logProbs.insert(logProbs.end(), other.logProbs.begin(), other.logProbs.end());
				actionMasks.insert(actionMasks.end(), other.actionMasks.begin(), other.actionMasks.end());
				terminals.insert(terminals.end(), other.terminals.begin(), other.terminals.end());
				actions.insert(actions.end(), other.actions.begin(), other.actions.end());
			}

			size_t Length() const {
				return actions.size();
			}
		};

		auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
		int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

		// Pré-allouer les vecteurs réutilisés
		std::vector<int> newPlayerIndicesReusable;
		std::vector<int> oldPlayerIndicesReusable;
		std::vector<bool> oldVersionPlayerMaskReusable;
		newPlayerIndicesReusable.reserve(numPlayers);
		oldPlayerIndicesReusable.reserve(numPlayers);
		oldVersionPlayerMaskReusable.reserve(numPlayers);
		
		Trajectory combinedTrajReusable;
		combinedTrajReusable.Reserve(config.ppo.tsPerItr * 2);

		// OPTIMISATION MAJEURE: Double buffer pour pipeline CPU/GPU
		// Pendant que le GPU traite le batch N, le CPU prépare le batch N+1
		torch::Tensor tStatesBuffer[2], tActionMasksBuffer[2];
		torch::Tensor tdStatesBuffer[2], tdActionMasksBuffer[2];
		int currentBuffer = 0;
		
		// OPTIMISATION: Pré-allouer les tenseurs GPU pour les indices (évite réallocation)
		torch::Tensor tNewPlayerIndicesGPU, tOldPlayerIndicesGPU;

		while (true) {
			Report report = {};

			bool isFirstIteration = (totalTimesteps == 0);

			GGL::PolicyVersion* oldVersion = NULL;
			newPlayerIndicesReusable.clear();
			oldPlayerIndicesReusable.clear();
			oldVersionPlayerMaskReusable.clear();
			
			torch::Tensor tNewPlayerIndices, tOldPlayerIndices;

			for (int i = 0; i < numPlayers; i++)
				newPlayerIndicesReusable.push_back(i);

			if (config.trainAgainstOldVersions) {
				RG_ASSERT(config.trainAgainstOldChance >= 0 && config.trainAgainstOldChance <= 1);
				bool shouldTrainAgainstOld =
					(RocketSim::Math::RandFloat() < config.trainAgainstOldChance)
					&& !versionMgr->versions.empty()
					&& !render;

				if (shouldTrainAgainstOld) {
					int oldVersionIdx = RocketSim::Math::RandInt(0, versionMgr->versions.size());
					oldVersion = &versionMgr->versions[oldVersionIdx];

					Team oldVersionTeam = Team(RocketSim::Math::RandInt(0, 2)); 
					
					newPlayerIndicesReusable.clear();
					oldVersionPlayerMaskReusable.resize(numPlayers);
					int i = 0;
					for (auto& state : envSet->state.gameStates) {
						for (auto& player : state.players) {
							if (player.team == oldVersionTeam) {
								oldVersionPlayerMaskReusable[i] = true;
								oldPlayerIndicesReusable.push_back(i);
							} else {
								oldVersionPlayerMaskReusable[i] = false;
								newPlayerIndicesReusable.push_back(i);
							}
							i++;
						}
					}

					tNewPlayerIndices = torch::tensor(newPlayerIndicesReusable);
					tOldPlayerIndices = torch::tensor(oldPlayerIndicesReusable);
					
					// OPTIMISATION: Pré-transférer les indices sur GPU une seule fois
					if (ppo->device.is_cuda()) {
						tNewPlayerIndicesGPU = tNewPlayerIndices.to(ppo->device, /*non_blocking=*/true);
						tOldPlayerIndicesGPU = tOldPlayerIndices.to(ppo->device, /*non_blocking=*/true);
					}
				}
			}

			int numRealPlayers = oldVersion ? newPlayerIndicesReusable.size() : envSet->state.numPlayers;

			int stepsCollected = 0;
			{ // Generate experience
				combinedTrajReusable.Clear();
				auto& combinedTraj = combinedTrajReusable;
				combinedTraj.Reserve(config.ppo.tsPerItr * 2);
				
				auto sanitizeActions = [&](std::vector<int>& actsVec) {
					bool clamped = false;
					for (int& a : actsVec) {
						if (a < 0) { a = 0; clamped = true; }
						else if (a >= numActions) { a = numActions - 1; clamped = true; }
					}
					if (clamped) {
						RG_LOG("Warning: clamped out-of-range action to valid bounds");
					}
				};

				Timer collectionTimer = {};
				{ // Collect timesteps
					RG_INFERENCE_MODE;

					float inferTime = 0;
					float envStepTime = 0;
					
					std::vector<int> curActionsVec;
					curActionsVec.reserve(numPlayers);
					FList newLogProbs;
					newLogProbs.reserve(numPlayers);
					std::vector<uint8_t> curTerminals(numPlayers, 0);

					auto& newPlayerIndices = newPlayerIndicesReusable;

					// OPTIMISATION MAJEURE: Future pour le travail GPU asynchrone
					std::future<void> gpuTransferFuture;
					bool hasGpuTransferPending = false;

					for (int step = 0; combinedTraj.Length() < config.ppo.tsPerItr || render; step++, stepsCollected += numRealPlayers) {
						Timer stepTimer = {};
						
						// OPTIMISATION: Lancer le reset des environnements en parallèle
						envSet->Reset();
						envStepTime += stepTimer.Elapsed();

#ifndef NDEBUG
						for (float f : envSet->state.obs.data)
							if (isnan(f) || isinf(f))
								RG_ERR_CLOSE("Obs builder produced a NaN/inf value");
#endif

						// OPTIMISATION: Normalisation in-place sur CPU (pendant que GPU fait autre chose)
						if (!render && obsStat) {
							int numSamples = RS_MIN(envSet->state.numPlayers, config.maxObsSamples);
							for (int i = 0; i < numSamples; i++) {
								int idx = Math::RandInt(0, envSet->state.numPlayers);
								obsStat->IncrementRow(&envSet->state.obs.At(idx, 0));
							}

							obsStat->NormalizeInPlace(
								envSet->state.obs.data.data(),
								envSet->state.numPlayers,
								obsSize,
								config.maxObsMeanRange,
								config.minObsSTD
							);
						}

						// OPTIMISATION: Créer les tenseurs CPU
						int bufIdx = currentBuffer;
						tStatesBuffer[bufIdx] = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
						tActionMasksBuffer[bufIdx] = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);

						// OPTIMISATION: Copier les obs dans les trajectoires EN PARALLÈLE avec le transfert GPU
						std::future<void> trajCopyFuture;
						if (!render) {
							trajCopyFuture = std::async(std::launch::async, [&, bufIdx]() {
								for (int newPlayerIdx : newPlayerIndices) {
									auto& traj = trajectories[newPlayerIdx];
									auto obsSpan = envSet->state.obs.GetRowSpan(newPlayerIdx);
									auto maskSpan = envSet->state.actionMasks.GetRowSpan(newPlayerIdx);
									traj.states.insert(traj.states.end(), obsSpan.begin(), obsSpan.end());
									traj.actionMasks.insert(traj.actionMasks.end(), maskSpan.begin(), maskSpan.end());
								}
							});
						}

						// OPTIMISATION: Lancer le transfert GPU de manière asynchrone
						if (ppo->device.is_cuda()) {
							GGL::GetStreamManager().RunOnTransferStream([&, bufIdx]() {
								tdStatesBuffer[bufIdx] = tStatesBuffer[bufIdx].to(ppo->device, /*non_blocking=*/true);
								tdActionMasksBuffer[bufIdx] = tActionMasksBuffer[bufIdx].to(ppo->device, /*non_blocking=*/true);
							});
						}

						// OPTIMISATION: Faire le step de l'environnement PENDANT le transfert GPU
						envSet->StepFirstHalf(true);

						// Attendre la copie des trajectoires
						if (!render && trajCopyFuture.valid()) {
							trajCopyFuture.wait();
						}

						Timer inferTimer = {};
						torch::Tensor tActions, tLogProbs;

						if (oldVersion) {
							if (ppo->device.is_cuda()) {
								GGL::GetStreamManager().WaitTransfers();
							}
							
							torch::Tensor srcStates = ppo->device.is_cuda() ? tdStatesBuffer[bufIdx] : tStatesBuffer[bufIdx];
							torch::Tensor srcMasks = ppo->device.is_cuda() ? tdActionMasksBuffer[bufIdx] : tActionMasksBuffer[bufIdx];
							
							// Utiliser les indices GPU pré-transférés
							torch::Tensor idxNew = ppo->device.is_cuda() ? tNewPlayerIndicesGPU : tNewPlayerIndices;
							torch::Tensor idxOld = ppo->device.is_cuda() ? tOldPlayerIndicesGPU : tOldPlayerIndices;
							
							torch::Tensor tdNewStates = srcStates.index_select(0, idxNew);
							torch::Tensor tdOldStates = srcStates.index_select(0, idxOld);
							torch::Tensor tdNewActionMasks = srcMasks.index_select(0, idxNew);
							torch::Tensor tdOldActionMasks = srcMasks.index_select(0, idxOld);
							
							if (!ppo->device.is_cuda()) {
								tdNewStates = tdNewStates.to(ppo->device, true);
								tdOldStates = tdOldStates.to(ppo->device, true);
								tdNewActionMasks = tdNewActionMasks.to(ppo->device, true);
								tdOldActionMasks = tdOldActionMasks.to(ppo->device, true);
							}

							torch::Tensor tNewActions;
							torch::Tensor tOldActions;

							ppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs);
							ppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);

							auto opts = torch::TensorOptions().dtype(tNewActions.dtype()).device(ppo->device);
							tActions = torch::zeros({ (int64_t)numPlayers }, opts);
							tActions.index_copy_(0, idxNew, tNewActions);
							tActions.index_copy_(0, idxOld, tOldActions);
							tActions = tActions.cpu();
						} else {
							if (ppo->device.is_cuda()) {
								GGL::GetStreamManager().WaitTransfers();
								ppo->InferActions(tdStatesBuffer[bufIdx], tdActionMasksBuffer[bufIdx], &tActions, &tLogProbs);
							} else {
								auto tdStates = tStatesBuffer[bufIdx].to(ppo->device, true);
								auto tdActionMasks = tActionMasksBuffer[bufIdx].to(ppo->device, true);
								ppo->InferActions(tdStates, tdActionMasks, &tActions, &tLogProbs);
							}
							tActions = tActions.cpu();
						}
						inferTime += inferTimer.Elapsed();

						// Alterner le buffer pour le prochain step
						currentBuffer = 1 - currentBuffer;

						TENSOR_TO_VEC_INPLACE<int>(tActions, curActionsVec);
						sanitizeActions(curActionsVec);
						
						if (tLogProbs.defined() && !render) {
							TENSOR_TO_VEC_INPLACE<float>(tLogProbs, newLogProbs);
						}

						stepTimer.Reset();
						envSet->Sync();
						envSet->StepSecondHalf(curActionsVec, false);
						envStepTime += stepTimer.Elapsed();

						if (stepCallback)
							stepCallback(this, envSet->state.gameStates, report);

						if (render) {
							renderSender->Send(envSet->state.gameStates[0]);
							continue;
						}

						// Calc average rewards (moins fréquent pour réduire overhead)
						if (config.addRewardsToMetrics && (Math::RandInt(0, config.rewardSampleRandInterval) == 0)) {
							int numSamples = RS_MIN(envSet->arenas.size(), config.maxRewardSamples);
							std::unordered_map<std::string, AvgTracker> avgRewards = {};
							for (int i = 0; i < numSamples; i++) {
								int arenaIdx = Math::RandInt(0, envSet->arenas.size());
								auto& prevRewards = envSet->state.lastRewards[i];

							 for (int j = 0; j < envSet->rewards[arenaIdx].size(); j++) {
								 std::string rewardName = envSet->rewards[arenaIdx][j].reward->GetName();
								 avgRewards[rewardName] += prevRewards[j];
							 }
						 }

						 for (auto& pair : avgRewards)
							 report.AddAvg("Rewards/" + pair.first, pair.second.Get());
						}

						// Ajouter aux trajectoires
						int i = 0;
						for (int newPlayerIdx : newPlayerIndices) {
							auto& traj = trajectories[newPlayerIdx];
							traj.actions.push_back(curActionsVec[newPlayerIdx]);
							traj.rewards.push_back(envSet->state.rewards[newPlayerIdx]);
							traj.logProbs.push_back(newLogProbs[i]);
							i++;
						}

						std::fill(curTerminals.begin(), curTerminals.end(), 0);
						for (int idx = 0; idx < envSet->arenas.size(); idx++) {
							uint8_t terminalType = envSet->state.terminals[idx];
							if (!terminalType)
								continue;

							auto playerStartIdx = envSet->state.arenaPlayerStartIdx[idx];
							int playersInArena = envSet->state.gameStates[idx].players.size();
							for (int i = 0; i < playersInArena; i++)
								curTerminals[playerStartIdx + i] = terminalType;
						}

						for (int newPlayerIdx : newPlayerIndices) {
						 int8_t terminalType = curTerminals[newPlayerIdx];
						 auto& traj = trajectories[newPlayerIdx];

						 if (!terminalType && traj.Length() >= maxEpisodeLength) {
							 terminalType = RLGC::TerminalType::TRUNCATED;
						 }

						 traj.terminals.push_back(terminalType);
						 if (terminalType) {

							 if (terminalType == RLGC::TerminalType::TRUNCATED) {
								 auto obsSpan = envSet->state.obs.GetRowSpan(newPlayerIdx);
								 traj.nextStates.insert(traj.nextStates.end(), obsSpan.begin(), obsSpan.end());
							 }

							 combinedTraj.Append(traj);
							 traj.Clear();
						 }
						}
					}

					report["Inference Time"] = inferTime;
					report["Env Step Time"] = envStepTime;
				}
				float collectionTime = collectionTimer.Elapsed();

				Timer consumptionTimer = {};
				{ // Process timesteps
					RG_INFERENCE_MODE;

					// OPTIMISATION MAJEURE: Créer tous les tenseurs en parallèle sur CPU
					torch::Tensor tStates, tActionMasks, tActions, tLogProbs, tRewards, tTerminals;
					
					std::atomic<int> tensorsCreated{0};
					
					// OPTIMISATION: Utiliser le ThreadPool pour créer les tenseurs en parallèle
					RLGC::g_ThreadPool.StartJobAsync([&]() {
						tActionMasks = GGL::VectorToTensor<uint8_t>(combinedTraj.actionMasks, { (int64_t)combinedTraj.actionMasks.size() / numActions, (int64_t)numActions });
						tensorsCreated++;
					});
					RLGC::g_ThreadPool.StartJobAsync([&]() {
						tActions = GGL::VectorToTensor<int32_t>(combinedTraj.actions, { (int64_t)combinedTraj.actions.size() });
						tensorsCreated++;
					});
					RLGC::g_ThreadPool.StartJobAsync([&]() {
						tLogProbs = GGL::VectorToTensor<float>(combinedTraj.logProbs, { (int64_t)combinedTraj.logProbs.size() });
						tensorsCreated++;
					});
					RLGC::g_ThreadPool.StartJobAsync([&]() {
						tRewards = GGL::VectorToTensor<float>(combinedTraj.rewards, { (int64_t)combinedTraj.rewards.size() });
						tensorsCreated++;
					});
					RLGC::g_ThreadPool.StartJobAsync([&]() {
						tTerminals = GGL::VectorToTensor<int8_t>(combinedTraj.terminals, { (int64_t)combinedTraj.terminals.size() });
						tensorsCreated++;
					});
					
					// Le plus gros dans le thread courant
					tStates = GGL::VectorToTensor<float>(combinedTraj.states, { (int64_t)combinedTraj.states.size() / obsSize, (int64_t)obsSize });
					tensorsCreated++;
					
					while (tensorsCreated.load() < 6) {
						std::this_thread::yield();
					}

					torch::Tensor tNextTruncStates;
					if (!combinedTraj.nextStates.empty())
						tNextTruncStates = GGL::VectorToTensor<float>(combinedTraj.nextStates, { (int64_t)combinedTraj.nextStates.size() / obsSize, (int64_t)obsSize });

					report["Average Step Reward"] = tRewards.mean().item<float>();
					report["Collected Timesteps"] = stepsCollected;
					
					// OPTIMISATION MAJEURE: Lancer le transfert GPU ET le calcul GAE en parallèle
					// GAE est sur CPU, donc on peut le faire pendant que les données sont transférées
					torch::Tensor tValPreds;
					torch::Tensor tTruncValPreds;
					torch::Tensor tAdvantages, tTargetVals, tReturns;
					float rewClipPortion = 0;

					std::future<void> gaeFuture;

					if (ppo->device.is_cpu()) {
						tValPreds = ppo->InferCritic(tStates.to(ppo->device, /*non_blocking=*/true, /*copy=*/true)).cpu();
						if (tNextTruncStates.defined())
							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, /*non_blocking=*/true, /*copy=*/true)).cpu();
						
						// GAE sur le thread courant
						Timer gaeTimer = {};
						GAE::Compute(
							tRewards, tTerminals, tValPreds, tTruncValPreds,
							tAdvantages, tTargetVals, tReturns, rewClipPortion,
							config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
						);
						report["GAE Time"] = gaeTimer.Elapsed();
					} else {
						// OPTIMISATION: GPU inference avec pipeline
						tValPreds = ppo->InferCriticBatched(tStates, ppo->config.miniBatchSize).cpu();
						
						if (tNextTruncStates.defined()) {
							tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(ppo->device, /*non_blocking=*/true, /*copy=*/true)).cpu();
						}
						
						// OPTIMISATION: GAE sur CPU en parallèle (les valPreds sont déjà sur CPU)
						Timer gaeTimer = {};
						GAE::Compute(
							tRewards, tTerminals, tValPreds, tTruncValPreds,
							tAdvantages, tTargetVals, tReturns, rewClipPortion,
							config.ppo.gaeGamma, config.ppo.gaeLambda, returnStat ? returnStat->GetSTD() : 1, config.ppo.rewardClipRange
						);
						report["GAE Time"] = gaeTimer.Elapsed();
					}

					report["Clipped Reward Portion"] = rewClipPortion;

					if (returnStat) {
						report["GAE/Returns STD"] = returnStat->GetSTD();

						int numToIncrement = RS_MIN(config.maxReturnSamples, (int)tReturns.size(0));
						if (numToIncrement > 0) {
							auto selectedReturns = tReturns.index_select(0, torch::randint(tReturns.size(0), { (int64_t)numToIncrement }));
							returnStat->Increment(TENSOR_TO_VEC<float>(selectedReturns));
						}
					}
					report["GAE/Avg Return"] = tReturns.abs().mean().item<float>();
					report["GAE/Avg Advantage"] = tAdvantages.abs().mean().item<float>();
					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();

					report["Episode Length"] = 1.f / (tTerminals == 1).to(torch::kFloat32).mean().item<float>();

					// Set experience buffer
					experience.data.actions = tActions;
					experience.data.logProbs = tLogProbs;
					experience.data.actionMasks = tActionMasks;
					experience.data.states = tStates;
					experience.data.advantages = tAdvantages;
					experience.data.targetValues = tTargetVals;
				}

				// Learn
				Timer learnTimer = {};
				ppo->Learn(experience, report, isFirstIteration);
				report["PPO Learn Time"] = learnTimer.Elapsed();

				// Set metrics
				float consumptionTime = consumptionTimer.Elapsed();
				report["Collection Time"] = collectionTime;
			 report["Consumption Time"] = consumptionTime;
			 report["Collection Steps/Second"] = stepsCollected / collectionTime;
			 report["Consumption Steps/Second"] = stepsCollected / consumptionTime;
			 report["Overall Steps/Second"] = stepsCollected / (collectionTime + consumptionTime);

				uint64_t prevTimesteps = totalTimesteps;
				totalTimesteps += stepsCollected;
				report["Total Timesteps"] = totalTimesteps;
				totalIterations++;
				report["Total Iterations"] = totalIterations;

				if (versionMgr)
					versionMgr->OnIteration(ppo, report, totalTimesteps, prevTimesteps);

				if (saveQueued) {
					if (!config.checkpointFolder.empty())
						Save();
					exit(0);
				}

				if (!config.checkpointFolder.empty()) {
					if (totalTimesteps / config.tsPerSave > prevTimesteps / config.tsPerSave) {
						Save();
					}
				}

				report.Finish();

				if (metricSender)
					metricSender->Send(report);

				report.Display(
					{
						"Average Step Reward",
						"Policy Entropy",
						"KL Div Loss",
						"First Accuracy",
						"",
						"Policy Update Magnitude",
						"Critic Update Magnitude",
						"Shared Head Update Magnitude",
						"",
						"Collection Steps/Second",
						"Consumption Steps/Second",
						"Overall Steps/Second",
						"",
						"Collection Time",
						"-Inference Time",
						"-Env Step Time",
						"Consumption Time",
						"-GAE Time",
						"-PPO Learn Time",
						"",
						"Collected Timesteps",
						"Total Timesteps",
						"Total Iterations"
					}
				);
			}
		}

		
	} catch (std::exception& e) {
		RG_ERR_CLOSE("Exception thrown during main learner loop: " << e.what());
	}
}

GGL::Learner::~Learner() {
	delete ppo;
	delete versionMgr;
	delete metricSender;
	delete renderSender;
	delete envSet;       // FIX: Libérer envSet
	delete returnStat;   // FIX: Libérer returnStat
	delete obsStat;      // FIX: Libérer obsStat
	if (ownsInterpreter && Py_IsInitialized())
		pybind11::finalize_interpreter();
}
