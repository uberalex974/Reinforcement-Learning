#include "../GigaLearnCPP/src/private/GigaLearnCPP/FrameworkTorch.h"



#include <GigaLearnCPP/Learner.h>


#include <RLGymCPP/Rewards/KickoffProximityReward2v2Enhanced.h>

#include <RLGymCPP/Rewards/CommonRewards.h>

#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>



#include <RLGymCPP/OBSBuilders/DefaultObs.h>

#include <RLGymCPP/OBSBuilders/AdvancedObs.h>

#include <RLGymCPP/StateSetters/KickoffState.h>

#include <RLGymCPP/StateSetters/RandomState.h>

#include <RLGymCPP/ActionParsers/DefaultAction.h>



#include <iostream>

#include <string>

#include <cmath>

#include <thread>



using namespace GGL; // GigaLearn

using namespace RLGC; // RLGymCPP

// Condition terminale: le match s'arrête quand une équipe atteint 'limit' buts.
// Maintient un compteur interne car GameState n'expose pas directement les scores.
class ScoreLimitCondition : public TerminalCondition {
public:
	explicit ScoreLimitCondition(int limitGoals) : limit(limitGoals), blueScore(0), orangeScore(0) {}
	virtual ~ScoreLimitCondition() = default;

	virtual void Reset(const GameState& initialState) override {
		blueScore = 0;
		orangeScore = 0;
	}

	virtual bool IsTerminal(const GameState& currentState) override {
		if (currentState.goalScored) {
			// Détermine quelle équipe a marqué en fonction de la position Y de la balle
			// Si la balle est dans le but orange (Y > 0), c'est un but pour l'équipe bleue
			// Si la balle est dans le but bleu (Y < 0), c'est un but pour l'équipe orange
			if (currentState.ball.pos.y > 0) {
				++blueScore;
			} else {
				++orangeScore;
			}
		}

		return (blueScore >= limit) || (orangeScore >= limit);
	}

	virtual bool IsTruncation() override {
		return false; // Fin de partie normale, pas une troncation
	}

	int GetBlueScore() const { return blueScore; }
	int GetOrangeScore() const { return orangeScore; }

private:
	int limit;
	int blueScore;
	int orangeScore;
};

// Pénalité continue pour les joueurs dont l'équipe est menée au score.
// Donne une pénalité proportionnelle à la différence de buts.
class LosingPenaltyReward : public Reward {
public:
	explicit LosingPenaltyReward(float penaltyPerGoalBehind = 0.01f) 
		: penaltyScale(penaltyPerGoalBehind), blueScore(0), orangeScore(0) {}

	virtual void Reset(const GameState& initialState) override {
		blueScore = 0;
		orangeScore = 0;
	}

	virtual void PreStep(const GameState& state) override {
		// Mise à jour du score si un but a été marqué
		if (state.goalScored) {
			if (state.ball.pos.y > 0) {
				++blueScore;
			} else {
				++orangeScore;
			}
		}
	}

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		int teamScore = (player.team == Team::BLUE) ? blueScore : orangeScore;
		int opponentScore = (player.team == Team::BLUE) ? orangeScore : blueScore;
		
		int deficit = opponentScore - teamScore;
		
		// Pénalité seulement si l'équipe est menée
		if (deficit > 0) {
			return -penaltyScale * static_cast<float>(deficit);
		}
		return 0.f;
	}

private:
	float penaltyScale;
	int blueScore;
	int orangeScore;
};

// Create the RLGymCPP environment for each of our games

EnvCreateResult EnvCreateFunc(int index) {

	// These are ok rewards that will produce a scoring bot in ~100m steps

	std::vector<WeightedReward> rewards = {

		// Movement

		{ new AirReward(), 0.25f },

		{ new WavedashReward(), 0.12f },
		
	    { new KickoffProximityReward2v2Enhanced(), 5.f  },
		
		// Player-ball
		
		//{ new FaceBallReward(), 0.01f },//

		{ new VelocityPlayerToBallReward(), 4.f },

		{ new StrongTouchReward(20, 120), 60 },

		{ new TouchAccelReward(), 6.f },

		// Ball-goal

		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1), 8.0f },



		// Boost

		{ new PickupBoostReward(), 0.1f },

		{ new SaveBoostReward(), 0.010f },



		// Game events

		{ new ZeroSumReward(new BumpReward(), 0.5f), 20 },

		{ new ZeroSumReward(new DemoReward(), 0.5f), 80 },

		{ new ZeroSumReward(new GoalReward(), 1), 150 },

		// Pénalité pour être mené au score (encourage les bots à remonter)
		{ new LosingPenaltyReward(0.02f), 1.0f }

	};

	// Conditions terminales (la limite de temps est gérée par cfg.ppo.maxEpisodeDuration)
	std::vector<TerminalCondition*> terminalConditions = {

		new NoTouchCondition(8),

		new ScoreLimitCondition(3) // épisode s'arrête quand une équipe atteint 3 buts

	};



	// Make the arena

	int playersPerTeam = 2;

	auto arena = Arena::Create(GameMode::SOCCAR);

	for (int i = 0; i < playersPerTeam; i++) {

		arena->AddCar(Team::BLUE);

		arena->AddCar(Team::ORANGE);

	}



	EnvCreateResult result = {};

	result.actionParser = new DefaultAction();

	result.obsBuilder = new AdvancedObs();

	result.stateSetter = new KickoffState();

	result.terminalConditions = terminalConditions;

	result.rewards = rewards;



	result.arena = arena;



	return result;

}



// Compteur thread-local pour optimiser le sampling des métriques (évite rand() répétitif)
static thread_local unsigned int stepCounter = 0;

void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {

	// Optimisation: utilise un compteur au lieu de rand() pour éviter le coût de génération aléatoire
	++stepCounter;
	bool doExpensiveMetrics = (stepCounter % 4) == 0;



	// Add our metrics

	for (const auto& state : states) {

		if (doExpensiveMetrics) {

			for (const auto& player : state.players) {

				report.AddAvg("Player/In Air Ratio", !player.isOnGround);

				report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);

				report.AddAvg("Player/Demoed Ratio", player.isDemoed);



				report.AddAvg("Player/Speed", player.vel.Length());

				Vec dirToBall = (state.ball.pos - player.pos).Normalized();

				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0, player.vel.Dot(dirToBall)));



				report.AddAvg("Player/Boost", player.boost);



				if (player.ballTouchedStep)

					report.AddAvg("Player/Touch Height", state.ball.pos.z);

			}

		}



		if (state.goalScored)

			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());

	}

}



int main(int argc, char* argv[]) try {
	std::ofstream _bootlog("startup.log", std::ios::app);
	_bootlog << "main entry\n";
	_bootlog.flush();
	std::cout << "GigaLearnBot starting..." << std::endl;

	// Optional fast-exit guard to help diagnose startup issues without running the full loop.
	if (std::getenv("GIGALEARN_QUICK_EXIT")) {
		std::cout << "GIGALEARN_QUICK_EXIT set, exiting early." << std::endl;
		return EXIT_SUCCESS;
	}
	// Détecter le mode (--render pour rendu, pas d'arguments = entrainement)

	bool renderMode = false;

	float scaleFactor = -1.0f; // if left negative, we'll auto-decide

	for (int i = 1; i < argc; ++i) {

		std::string arg = argv[i];

		if (arg == "--render") {

			renderMode = true;

			break;

		}

		// Support --scale=1.5 or --scale 1.5 (explicit override)

		if (arg.rfind("--scale=", 0) == 0) {

			try { scaleFactor = std::stof(arg.substr(8)); } catch(...) {}

		} else if (arg == "--scale" && i + 1 < argc) {

			try { scaleFactor = std::stof(argv[i+1]); } catch(...) {}

		}

	}



	if (renderMode)

		std::cout << "GigaLearn: démarrage en mode rendu (--render)\n";

	else

		std::cout << "GigaLearn: démarrage en mode entrainement (double-clique sans arguments)\n";
	


	// Initialize RocketSim with collision meshes

	// Change this path to point to your meshes

	RocketSim::Init("C:\\Giga\\GigaLearnCPP-Leak\\collision_meshes");



	// Make configuration for the learner

	LearnerConfig cfg = {};

	cfg.deviceType = LearnerDeviceType::GPU_CUDA;



	cfg.tickSkip = 8;

	cfg.actionDelay = cfg.tickSkip - 1; // Normal value in other RLGym frameworks



	// Play around with this to see what the optimal is for your machine, more games will consume more RAM

	cfg.numGames = 512;



	// Leave this empty to use a random seed each run

	// The random seed can have a strong effect on the outcome of a run

	cfg.randomSeed = 123;



	int tsPerItr = 100'000;

	cfg.ppo.tsPerItr = tsPerItr;

	cfg.ppo.batchSize = tsPerItr;

	cfg.ppo.miniBatchSize = 50'000; // Lower this if too much VRAM is being allocated

	
	// Limite de temps des épisodes: 5 minutes = 300 secondes
	cfg.ppo.maxEpisodeDuration = 300.0;

	// Using 2 epochs seems pretty optimal when comparing time training to skill

	// Perhaps 1 or 3 is better for you, test and find out!

	cfg.ppo.epochs = 2;



	// This scales differently than "ent_coef" in other frameworks

	// This is the scale for normalized entropy, which means you won't have to change it if you add more actions

	cfg.ppo.entropyScale = 0.035f;



	// Rate of reward decay

	// Starting low tends to work out

	cfg.ppo.gaeGamma = 0.99;



	// Good learning rate to start

	cfg.ppo.policyLR = 2.5e-4;

	cfg.ppo.criticLR = 2.5e-4;



	// Base sizes (current working configuration)

	std::vector<int> baseShared = { 512, 512 };

	std::vector<int> basePolicy = { 512, 512, 512 };

	std::vector<int> baseCritic = { 512, 512, 512 };



	// Auto-decide scale factor if not explicitly provided

	if (scaleFactor <= 0.f) {

		if (torch::cuda::is_available()) {

			int devCount = 0;

			try { devCount = torch::cuda::device_count(); } catch (...) { devCount = 1; }

			// Base GPU budget scaling

			scaleFactor = 1.4f; // conservative baseline for single GPU

			if (devCount >= 2) scaleFactor = 1.8f;

			if (devCount >= 4) scaleFactor = 2.2f;

			// Slightly increase if user configured many parallel games

			if (cfg.numGames >= 512) scaleFactor += 0.1f;

		} else {

			unsigned hc = std::thread::hardware_concurrency();

			if (hc == 0) hc = 4;

			if (hc >= 16) scaleFactor = 1.25f;

			else if (hc >= 8) scaleFactor = 1.15f;

			else scaleFactor = 1.05f;

		}

	}



	// Clamp reasonable bounds

	if (scaleFactor < 1.0f) scaleFactor = 1.0f;

	if (scaleFactor > 3.0f) scaleFactor = 3.0f;



	std::cout << "Model scale factor (auto): " << scaleFactor << "\n";



	// Scale sizes intelligently but conservatively

	auto scaleVec = [&](const std::vector<int>& base) {

		std::vector<int> out;

		for (int v : base) {

			int nv = std::max(1, (int)std::lround(v * scaleFactor));

			// keep values multiple of 8 for better GPU alignment

			if (nv % 8 != 0) nv += (8 - (nv % 8));

			out.push_back(nv);

		}

		return out;

	};



	auto newShared = scaleVec(baseShared);

	auto newPolicy = scaleVec(basePolicy);

	auto newCritic = scaleVec(baseCritic);



	cfg.ppo.sharedHead.layerSizes = newShared;

	cfg.ppo.policy.layerSizes = newPolicy;

	cfg.ppo.critic.layerSizes = newCritic;



	// Print resulting sizes for quick verification

	auto printVec = [&](const std::vector<int>& v) {

		std::cout << "[";

		for (size_t i = 0; i < v.size(); ++i) {

			std::cout << v[i];

			if (i + 1 < v.size()) std::cout << ", ";

		}

		std::cout << "]";

	};



	std::cout << "Shared head sizes: "; printVec(cfg.ppo.sharedHead.layerSizes); std::cout << "\n";

	std::cout << "Policy sizes: "; printVec(cfg.ppo.policy.layerSizes); std::cout << "\n";

	std::cout << "Critic sizes: "; printVec(cfg.ppo.critic.layerSizes); std::cout << "\n";



	auto optim = ModelOptimType::ADAMW;

	cfg.ppo.policy.optimType = optim;

	cfg.ppo.critic.optimType = optim;

	cfg.ppo.sharedHead.optimType = optim;



	auto activation = ModelActivationType::LEAKY_RELU;

	cfg.ppo.policy.activationType = activation;

	cfg.ppo.critic.activationType = activation;

	cfg.ppo.sharedHead.activationType = activation;



	bool addLayerNorm = true;

	cfg.ppo.policy.addLayerNorm = addLayerNorm;

	cfg.ppo.critic.addLayerNorm = addLayerNorm;

	cfg.ppo.sharedHead.addLayerNorm = addLayerNorm;



	cfg.sendMetrics = true; // Send metrics

	cfg.renderMode = renderMode; // <-- Use the flag parsed from the command line



	// Make the learner with the environment creation function and the config we just made

	Learner* learner = new Learner(EnvCreateFunc, cfg, StepCallback);



	// Start learning!

	learner->Start();



	return EXIT_SUCCESS;
} catch (const c10::Error& e) {
	std::cerr << "C10/Torch runtime error: " << e.what_without_backtrace() << std::endl;
	return EXIT_FAILURE;
} catch (const std::exception& e) {
	std::cerr << "Fatal error: " << e.what() << std::endl;
	return EXIT_FAILURE;
} catch (...) {
	std::cerr << "Fatal error: unknown exception." << std::endl;
	return EXIT_FAILURE;
}
