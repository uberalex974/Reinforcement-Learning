#include "ZeroSumReward.h"

std::vector<float> RLGC::ZeroSumReward::GetAllRewards(const GameState& state, bool final) {
	// OPTIMISATION: Réutiliser _lastRewards comme buffer de sortie
	const size_t numPlayers = state.players.size();
	
	// Récupérer les récompenses brutes
	std::vector<float> rawRewards = child->GetAllRewards(state, final);
	
	// Redimensionner _lastRewards si nécessaire
	if (_lastRewards.size() != numPlayers) {
		_lastRewards.resize(numPlayers);
	}
	
	// Copier les récompenses brutes pour le logging
	std::copy(rawRewards.begin(), rawRewards.end(), _lastRewards.begin());

	// OPTIMISATION: Calcul optimisé avec moins de passes sur les données
	int teamCounts[2] = {0, 0};
	float teamSums[2] = {0.0f, 0.0f};

	// Première passe: calculer les sommes et comptes par équipe
	for (size_t i = 0; i < numPlayers; i++) {
		const int teamIdx = static_cast<int>(state.players[i].team);
		teamCounts[teamIdx]++;
		teamSums[teamIdx] += rawRewards[i];
	}

	// Calculer les moyennes (éviter division par zéro)
	const float avgTeam0 = teamCounts[0] > 0 ? teamSums[0] / teamCounts[0] : 0.0f;
	const float avgTeam1 = teamCounts[1] > 0 ? teamSums[1] / teamCounts[1] : 0.0f;
	
	// Pré-calculer les coefficients
	const float selfCoef = 1.0f - teamSpirit;
	
	// Deuxième passe: appliquer la transformation
	for (size_t i = 0; i < numPlayers; i++) {
		const int teamIdx = static_cast<int>(state.players[i].team);
		const float avgTeamReward = teamIdx == 0 ? avgTeam0 : avgTeam1;
		const float avgOpponentReward = teamIdx == 0 ? avgTeam1 : avgTeam0;
		
		rawRewards[i] = 
			rawRewards[i] * selfCoef
			+ avgTeamReward * teamSpirit
			- avgOpponentReward * opponentScale;
	}
	
	return rawRewards;
}