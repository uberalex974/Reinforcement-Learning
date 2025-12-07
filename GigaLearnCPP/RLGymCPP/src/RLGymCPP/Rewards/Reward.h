#pragma once
#include "../Gamestates/GameState.h"
#include "../BasicTypes/Action.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/reward_function.py
namespace RLGC {
	class Reward {
	private:
		std::string _cachedName = {};
		
		// OPTIMISATION: Cache pour les récompenses calculées
		mutable std::vector<float> _rewardsCache;

	public:
		virtual void Reset(const GameState& initialState) {}

		virtual void PreStep(const GameState& state) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			throw std::runtime_error("GetReward() is unimplemented");
			return 0;
		}

		// OPTIMISATION MAJEURE: Version qui réutilise le buffer interne
		// Évite l'allocation de vecteur à chaque step
		virtual std::vector<float> GetAllRewards(const GameState& state, bool isFinal) {
			const size_t numPlayers = state.players.size();
			
			// Réutiliser le cache si possible
			if (_rewardsCache.size() != numPlayers) {
				_rewardsCache.resize(numPlayers);
			}
			
			for (size_t i = 0; i < numPlayers; i++) {
				_rewardsCache[i] = GetReward(state.players[i], state, isFinal);
			}

			return _rewardsCache;
		}
		
		// NOUVELLE FONCTIONNALITÉ: Version qui écrit directement dans un buffer externe
		// Évite complètement l'allocation et la copie
		virtual void GetAllRewardsInPlace(const GameState& state, bool isFinal, float* output) {
			const size_t numPlayers = state.players.size();
			for (size_t i = 0; i < numPlayers; i++) {
				output[i] = GetReward(state.players[i], state, isFinal);
			}
		}

		// Méthode virtuelle pour obtenir les récompenses internes sans dynamic_cast
		virtual const std::vector<float>* GetInnerRewards() const {
			return nullptr;
		}

		virtual std::string GetName() {

			if (!_cachedName.empty())
				return _cachedName;

			std::string rewardName = typeid(*this).name();

			// Trim the string to after cetain keys
			{
				constexpr const char* TRIM_KEYS[] = {
					"::", // Namespace separator
					" " // Any spaces
				};
				for (const char* key : TRIM_KEYS) {
					size_t idx = rewardName.rfind(key);
					if (idx == std::string::npos)
						continue;

					rewardName.erase(rewardName.begin(), rewardName.begin() + idx + strlen(key));
				}
			}

			_cachedName = rewardName;
			return rewardName;
		}

		virtual ~Reward() {};
	};

	struct WeightedReward {
		Reward* reward;
		float weight;

		WeightedReward(Reward* reward, float scale) : reward(reward), weight(scale) {}
		WeightedReward(Reward* reward, int scale) : reward(reward), weight(scale) {}
	};
}