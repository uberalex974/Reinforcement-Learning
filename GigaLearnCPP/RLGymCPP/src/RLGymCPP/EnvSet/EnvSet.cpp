#include "EnvSet.h"
#include  "../Rewards/ZeroSumReward.h"

template<bool RLGC::PlayerEventState::* DATA_VAR>
void IncPlayerCounter(Car* car, void* userInfoPtr) {
	if (!car)
		return;

	auto userInfo = (RLGC::EnvSet::CallbackUserInfo*)userInfoPtr;

	auto& gs = userInfo->envSet->state.gameStates[userInfo->arenaIdx];
	for (auto& player : gs.players)
		if (player.carId == car->id)
			(player.eventState.*DATA_VAR) = true;
}

void _ShotEventCallback(Arena* arena, Car* shooter, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::shot>(shooter, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::shotPass>(passer, userInfo);
}

void _GoalEventCallback(Arena* arena, Car* scorer, Car* passer, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::goal>(scorer, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::assist>(passer, userInfo);
}

void _SaveEventCallback(Arena* arena, Car* saver, void* userInfo) {
	IncPlayerCounter<&RLGC::PlayerEventState::save>(saver, userInfo);
}

void _BumpCallback(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo) {
	if (bumper->team == victim->team)
		return;

	IncPlayerCounter<&RLGC::PlayerEventState::bump>(bumper, userInfo);
	IncPlayerCounter<&RLGC::PlayerEventState::bumped>(victim, userInfo);

	if (isDemo) {
		IncPlayerCounter<&RLGC::PlayerEventState::demo>(bumper, userInfo);
		IncPlayerCounter<&RLGC::PlayerEventState::demoed>(victim, userInfo);
	}
}

/////////////////////////////

RLGC::EnvSet::EnvSet(const EnvSetConfig& config) : config(config) {

	RG_ASSERT(config.tickSkip > 0);
	RG_ASSERT(config.actionDelay >= 0 && config.actionDelay <= config.tickSkip);

	std::mutex appendMutex = {};
	auto fnCreateArenas = [&](int idx) {
		auto createResult = config.envCreateFn(idx);
		auto arena = createResult.arena;

		appendMutex.lock();
		{
			arenas.push_back(arena);

			auto userInfo = new CallbackUserInfo();
			userInfo->arena = arena;
			userInfo->arenaIdx = idx;
			userInfo->envSet = this;
			eventCallbackInfos.push_back(userInfo);
			arena->SetCarBumpCallback(_BumpCallback, userInfo);

			if (arena->gameMode != GameMode::HEATSEEKER) {
				GameEventTracker* tracker = new GameEventTracker({});
				eventTrackers.push_back(tracker);

				tracker->SetShotCallback(_ShotEventCallback, userInfo);
				tracker->SetGoalCallback(_GoalEventCallback, userInfo);
				tracker->SetSaveCallback(_SaveEventCallback, userInfo);
			} else {
				eventTrackers.push_back(NULL);
				eventCallbackInfos.push_back(NULL);
			}

			userInfos.push_back(createResult.userInfo);

			rewards.push_back(createResult.rewards);
			terminalConditions.push_back(createResult.terminalConditions);
			obsBuilders.push_back(createResult.obsBuilder);
			actionParsers.push_back(createResult.actionParser);
			stateSetters.push_back(createResult.stateSetter);
		}
		appendMutex.unlock();
	};
	g_ThreadPool.StartBatchedJobs(fnCreateArenas, config.numArenas, false);

	state.Resize(arenas);
	
	// Determine obs size and action amount, initialize arrays accordingly
	{
		stateSetters[0]->ResetArena(arenas[0]);
		GameState testState = GameState(arenas[0]);
		testState.userInfo = userInfos[0];
		obsBuilders[0]->Reset(testState);
		obsSize = obsBuilders[0]->BuildObs(testState.players[0], testState).size();
		state.obs = DimList2<float>(state.numPlayers, obsSize);

		state.actionMasks = DimList2<uint8_t>(state.numPlayers, actionParsers[0]->GetActionAmount());
	}

	// Reset all arenas initially
	g_ThreadPool.StartBatchedJobs(
		std::bind(&RLGC::EnvSet::ResetArena, this, std::placeholders::_1),
		config.numArenas, false
	);
	
}

void RLGC::EnvSet::StepFirstHalf(bool async) {

	auto fnStepArena = [&](int arenaIdx) {
		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];

		// Set previous gamestates
		state.prevGameStates[arenaIdx] = gs;

		gs.ResetBeforeStep();

		// Step arena with old actions
		arena->Step(config.actionDelay);
	};

	// OPTIMISATION: Utiliser chunked jobs pour réduire l'overhead du thread pool
	g_ThreadPool.StartBatchedJobsChunked(fnStepArena, arenas.size(), async);
}

void RLGC::EnvSet::StepSecondHalf(const IList& actionIndices, bool async) {

	auto fnStepArenas = [&](int arenaIdx) {

		Arena* arena = arenas[arenaIdx];
		auto& gs = state.gameStates[arenaIdx];
		const int playerStartIdx = state.arenaPlayerStartIdx[arenaIdx];
		const int numPlayersInArena = static_cast<int>(gs.players.size());
			
		// OPTIMISATION: thread_local pour éviter les allocations
		thread_local std::vector<Action> actions;
		actions.resize(numPlayersInArena);
		
		// Parse and set actions
		auto carItr = arena->_cars.begin();
		for (int i = 0; i < numPlayersInArena; i++, carItr++) {
			auto& player = gs.players[i];
			Car* car = *carItr;
			Action action = actionParsers[arenaIdx]->ParseAction(actionIndices[playerStartIdx + i], player, gs);
			car->controls = (CarControls)action;
			actions[i] = action;
		}

		// Step arena
		arena->Step(config.tickSkip - config.actionDelay);

		if (eventTrackers[arenaIdx])
			eventTrackers[arenaIdx]->Update(arena);

		GameState* gsPrev = &state.prevGameStates[arenaIdx];
		if (gsPrev->IsEmpty())
			gsPrev = NULL;

		gs.UpdateFromArena(arena, actions, gsPrev);

		// Update terminal
		uint8_t terminalType = TerminalType::NOT_TERMINAL;
		for (auto cond : terminalConditions[arenaIdx]) {
			if (cond->IsTerminal(gs)) {
				bool isTrunc = cond->IsTruncation();
				uint8_t curTerminalType = isTrunc ? TerminalType::TRUNCATED : TerminalType::NORMAL;
				if (terminalType == TerminalType::NOT_TERMINAL) {
					terminalType = curTerminalType;
				} else if (curTerminalType == TerminalType::NORMAL) {
					terminalType = curTerminalType;
				}
			}
		}
		state.terminals[arenaIdx] = terminalType;
		
		// Pre-step rewards
		for (auto& weighted : rewards[arenaIdx])
			weighted.reward->PreStep(gs);

		// OPTIMISATION MAJEURE: Réutiliser allRewards avec thread_local
		thread_local FList allRewards;
		allRewards.assign(numPlayersInArena, 0.0f);
		
		// OPTIMISATION: Cache le nombre de reward functions
		const int numRewardFuncs = static_cast<int>(rewards[arenaIdx].size());
		
		// OPTIMISATION: Pré-allouer lastRewards si nécessaire
		if (config.saveRewards && state.lastRewards[arenaIdx].size() != static_cast<size_t>(numRewardFuncs)) {
			state.lastRewards[arenaIdx].resize(numRewardFuncs);
		}
		
		// OPTIMISATION MAJEURE: Buffer thread-local pour éviter allocation par reward
		thread_local FList rewardOutputBuffer;
		rewardOutputBuffer.resize(numPlayersInArena);
		
		for (int rewardIdx = 0; rewardIdx < numRewardFuncs; rewardIdx++) {
			auto& weightedReward = rewards[arenaIdx][rewardIdx];
			
			// OPTIMISATION: Utiliser GetAllRewardsInPlace pour éviter l'allocation
			weightedReward.reward->GetAllRewardsInPlace(gs, terminalType, rewardOutputBuffer.data());
			
			const float weight = weightedReward.weight;
			
			// OPTIMISATION: Accès direct aux données sans bounds checking
			float* allRewardsPtr = allRewards.data();
			const float* outputPtr = rewardOutputBuffer.data();
			
			// OPTIMISATION: Loop unrolling x4 pour 2v2 (4 joueurs)
			int i = 0;
			const int unrollEnd = numPlayersInArena - (numPlayersInArena % 4);
			for (; i < unrollEnd; i += 4) {
				allRewardsPtr[i]   += outputPtr[i]   * weight;
				allRewardsPtr[i+1] += outputPtr[i+1] * weight;
				allRewardsPtr[i+2] += outputPtr[i+2] * weight;
				allRewardsPtr[i+3] += outputPtr[i+3] * weight;
			}
			for (; i < numPlayersInArena; i++) {
				allRewardsPtr[i] += outputPtr[i] * weight;
			}

			if (config.saveRewards) {
				int playerSampleIndex;
				if (config.shuffleRewardSampling) {
					playerSampleIndex = Math::RandInt(0, numPlayersInArena);
				} else {
					playerSampleIndex = 0;
					int lowestID = gs.players[0].carId;
					for (int pi = 1; pi < numPlayersInArena; pi++) {
						if (gs.players[pi].carId < lowestID) {
							lowestID = gs.players[pi].carId;
							playerSampleIndex = pi;
						}
					}
				}
				float rewardToSave = rewardOutputBuffer[playerSampleIndex];
					
				const std::vector<float>* innerRewards = weightedReward.reward->GetInnerRewards();
				if (innerRewards && playerSampleIndex < static_cast<int>(innerRewards->size())) {
					rewardToSave = (*innerRewards)[playerSampleIndex];
				}

				state.lastRewards[arenaIdx][rewardIdx] = rewardToSave;
			}
		}

		// OPTIMISATION: Copie directe des rewards
		for (int i = 0; i < numPlayersInArena; i++) {
			state.rewards[playerStartIdx + i] = allRewards[i];
		}

		// OPTIMISATION MAJEURE: Build obs et masks en utilisant SetFromPtr quand possible
		for (int i = 0; i < numPlayersInArena; i++) {
			const auto& player = gs.players[i];
			
			// Build obs et set directement
			auto obsVec = obsBuilders[arenaIdx]->BuildObs(player, gs);
			state.obs.SetFromPtr(playerStartIdx + i, obsVec.data(), obsVec.size());
			
			// Build action mask et set directement
			auto maskVec = actionParsers[arenaIdx]->GetActionMask(player, gs);
			state.actionMasks.SetFromPtr(playerStartIdx + i, maskVec.data(), maskVec.size());
		}
	};

	// OPTIMISATION: Utiliser chunked jobs pour réduire l'overhead
	g_ThreadPool.StartBatchedJobsChunked(fnStepArenas, arenas.size(), async);
}

void RLGC::EnvSet::ResetArena(int index) {
	stateSetters[index]->ResetArena(arenas[index]);
	GameState newState = GameState(arenas[index]);
	state.gameStates[index] = newState;

	newState.userInfo = userInfos[index];

	if (eventTrackers[index])
		eventTrackers[index]->ResetPersistentInfo();

	obsBuilders[index]->Reset(newState);
	for (auto& cond : terminalConditions[index])
		cond->Reset(newState);
	for (auto& weightedReward : rewards[index])
		weightedReward.reward->Reset(newState);

	const int playerStartIdx = state.arenaPlayerStartIdx[index];
	const int numPlayers = static_cast<int>(newState.players.size());
	
	// OPTIMISATION: Build obs and masks using SetFromPtr
	for (int i = 0; i < numPlayers; i++) {
		auto obsVec = obsBuilders[index]->BuildObs(newState.players[i], newState);
		state.obs.SetFromPtr(playerStartIdx + i, obsVec.data(), obsVec.size());
		
		auto maskVec = actionParsers[index]->GetActionMask(newState.players[i], newState);
		state.actionMasks.SetFromPtr(playerStartIdx + i, maskVec.data(), maskVec.size());
	}

	state.prevGameStates[index].MakeEmpty();
}

void RLGC::EnvSet::Reset() {
	// OPTIMISATION: Early exit si rien à réinitialiser
	bool hasTerminals = false;
	const size_t numArenas = arenas.size();
	
	for (size_t i = 0; i < numArenas; i++) {
		if (state.terminals[i]) {
			hasTerminals = true;
			break;
		}
	}
	
	if (!hasTerminals) {
		return;
	}
	
	// OPTIMISATION: thread_local vector pour éviter réallocation
	thread_local std::vector<int> indicesToReset;
	indicesToReset.clear();
	indicesToReset.reserve(numArenas);
	
	for (size_t i = 0; i < numArenas; i++) {
		if (state.terminals[i]) {
			indicesToReset.push_back(static_cast<int>(i));
		}
	}
	
	// Reset terminals immediately (AVANT les resets pour éviter double-reset)
	for (int idx : indicesToReset) {
		state.terminals[idx] = 0;
	}
	
	// OPTIMISATION: Parallel reset si plusieurs arènes à réinitialiser
	const size_t numToReset = indicesToReset.size();
	if (numToReset > 2) {
		// Utiliser le thread pool pour les resets parallèles
		for (int idx : indicesToReset) {
			g_ThreadPool.StartJobAsync([this, idx]() {
				ResetArena(idx);
			});
		}
		g_ThreadPool.WaitUntilDone();
	} else {
		// Pour 1-2 arènes, le séquentiel est plus rapide (overhead du pool)
		for (int idx : indicesToReset) {
			ResetArena(idx);
		}
	}
}