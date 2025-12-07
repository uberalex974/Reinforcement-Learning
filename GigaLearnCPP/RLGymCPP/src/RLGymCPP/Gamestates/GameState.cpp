#include "GameState.h"

#include "../Math.h"

using namespace RLGC;

static int boostPadIndexMap[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
static bool boostPadIndexMapBuilt = false;
static std::mutex boostPadIndexMapMutex = {};

void _BuildBoostPadIndexMap(Arena* arena) {
	constexpr const char* ERROR_PREFIX = "_BuildBoostPadIndexMap(): ";
#ifdef RG_VERBOSE
	RG_LOG("Building boost pad index map...");
#endif

	if (arena->_boostPads.size() != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		RG_ERR_CLOSE(
			ERROR_PREFIX << "Arena boost pad count does not match CommonValues::BOOST_LOCATIONS_AMOUNT " <<
			"(" << arena->_boostPads.size() << "/" << CommonValues::BOOST_LOCATIONS_AMOUNT << ")"
		);
	}
	
	bool found[CommonValues::BOOST_LOCATIONS_AMOUNT] = {};
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		Vec targetPos = CommonValues::BOOST_LOCATIONS[i];
		for (int j = 0; j < arena->_boostPads.size(); j++) {
			Vec padPos = arena->_boostPads[j]->config.pos;

			if (padPos.DistSq2D(targetPos) < 10) {
				if (!found[i]) {
					found[i] = true;
					boostPadIndexMap[i] = j;
				} else {
					RG_ERR_CLOSE(
						ERROR_PREFIX << "Matched duplicate boost pad at " << targetPos << "=" << padPos
					);
				}
				break;
			}
		}

		if (!found[i])
			RS_ERR_CLOSE(ERROR_PREFIX << "Failed to find matching pad at " << targetPos);
	}

#ifdef RG_VERBOSE
	RG_LOG(" > Done");
#endif
	boostPadIndexMapBuilt = true;
}

void RLGC::GameState::ResetBeforeStep() {
	// OPTIMISATION: Utiliser une boucle optimisée pour réinitialiser les events
	for (auto& player : players) {
		player.eventState = PlayerEventState{};
	}
}

void RLGC::GameState::UpdateFromArena(Arena* arena, const std::vector<Action>& actions, GameState* prev) {
	this->prev = prev;
	if (prev)
		prev->prev = NULL;

	lastArena = arena;
	
	// OPTIMISATION: Utiliser des entiers au lieu de uint64_t quand possible
	const uint64_t currentTick = arena->tickCount;
	const int tickSkip = static_cast<int>(RS_MAX(currentTick - lastTickCount, 0ULL));
	deltaTime = tickSkip * (1.0f / 120.0f);

	// OPTIMISATION: Copie directe du ball state
	ball = arena->ball->GetState();

	// OPTIMISATION: Éviter le resize si même taille
	const size_t numCars = arena->_cars.size();
	if (players.size() != numCars) {
		players.resize(numCars);
	}

	// OPTIMISATION: Utiliser un itérateur et un index en parallèle
	auto carItr = arena->_cars.begin();
	for (size_t i = 0; i < numCars; i++, ++carItr) {
		auto& player = players[i];
		player.index = static_cast<int>(i);
		player.UpdateFromCar(*carItr, currentTick, tickSkip, actions[i], prev ? &prev->players[i] : NULL);
		if (player.ballTouchedStep)
			lastTouchCarID = player.carId;
	}

	// Construction paresseuse de l'index map (une seule fois)
	if (!boostPadIndexMapBuilt) {
		boostPadIndexMapMutex.lock();
		if (!boostPadIndexMapBuilt) 
			_BuildBoostPadIndexMap(arena);
		boostPadIndexMapMutex.unlock();
	}

	// OPTIMISATION MAJEURE: Traitement vectorisé des boost pads
	const int numBoostPads = static_cast<int>(arena->_boostPads.size());
	
	// Redimensionner seulement si nécessaire
	if (boostPads.size() != static_cast<size_t>(numBoostPads)) {
		boostPads.resize(numBoostPads);
		boostPadsInv.resize(numBoostPads);
		boostPadTimers.resize(numBoostPads);
		boostPadTimersInv.resize(numBoostPads);
	}
	
	// OPTIMISATION: Pré-calculer les indices inversés et traiter par paires
	// Cela exploite mieux le cache CPU
	for (int i = 0; i < numBoostPads; i++) {
		const int idx = boostPadIndexMap[i];
		const int invI = CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1;
		const int invIdx = boostPadIndexMap[invI];

		const auto& state = arena->_boostPads[idx]->GetState();
		const auto& stateInv = arena->_boostPads[invIdx]->GetState();

		boostPads[i] = state.isActive;
		boostPadsInv[i] = stateInv.isActive;

		boostPadTimers[i] = state.cooldown;
		boostPadTimersInv[i] = stateInv.cooldown;
	}

	// Update goal scoring
	goalScored = arena->IsBallScored();

	lastTickCount = currentTick;
}
