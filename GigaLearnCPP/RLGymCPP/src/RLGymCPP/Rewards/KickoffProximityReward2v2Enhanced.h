#pragma once
#include "Reward.h"
#include "../CommonValues.h"

namespace RLGC {

class KickoffProximityReward2v2Enhanced : public Reward {
public:
	float goerReward = 1.2f;           // Increased base reward for goer
	float cheaterReward = 0.6f;        // Base reward for strategic cheater
	float dynamicWeight = 0.3f;        // Weight for dynamic adjustments
	float rotationPrepWeight = 0.2f;   // Weight for rotation preparation

	virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		// Enhanced kickoff detection - more robust
		if (!IsKickoffActive(state)) return 0.f;

		float playerDistToBall = (player.pos - state.ball.pos).Length();

		// Enhanced team analysis
		TeamAnalysis analysis = AnalyzeTeamState(player, state);
		if (!analysis.hasTeammate) return 0.f;

		// Dynamic role assignment with multiple factors
		PlayerRole role = DeterminePlayerRole(player, analysis, state);

		if (role == PlayerRole::GOER) {
			return CalculateGoerReward(player, analysis, state);
		}
		else {
			return CalculateCheaterReward(player, analysis, state);
		}
	}

private:
	enum class PlayerRole { GOER, CHEATER };

	struct TeamAnalysis {
		bool hasTeammate = false;
		const Player* teammate = nullptr;
		float teammateDistToBall = 0.f;
		float closestOpponentDist = FLT_MAX;
		float secondOpponentDist = FLT_MAX;
		Vec opponentCenterOfMass = Vec(0.f, 0.f, 0.f);  // Fixed: explicit float construction
		float avgOpponentSpeed = 0.f;
	};

	bool IsKickoffActive(const GameState& state) {
		// More sophisticated kickoff detection
		float ballSpeed = state.ball.vel.Length();
		float ballHeight = state.ball.pos.z;
		Vec ballPos2D = Vec(state.ball.pos.x, state.ball.pos.y, 0.f);  // Fixed: explicit float for z

		return (ballSpeed < 2.f &&
			ballHeight < 150.f &&
			ballPos2D.Length() < 50.f);
	}

	TeamAnalysis AnalyzeTeamState(const Player& player, const GameState& state) {
		TeamAnalysis analysis = {};  // Fixed: explicit initialization
		int opponentCount = 0;
		float totalOpponentSpeed = 0.f;

		for (const auto& p : state.players) {
			if (p.team == player.team && p.carId != player.carId) {
				analysis.teammate = &p;
				analysis.hasTeammate = true;
				analysis.teammateDistToBall = (p.pos - state.ball.pos).Length();
			}
			else if (p.team != player.team) {
				float opponentDist = (p.pos - state.ball.pos).Length();
				totalOpponentSpeed += p.vel.Length();
				opponentCount++;

				if (opponentDist < analysis.closestOpponentDist) {
					analysis.secondOpponentDist = analysis.closestOpponentDist;
					analysis.closestOpponentDist = opponentDist;
				}
				else if (opponentDist < analysis.secondOpponentDist) {
					analysis.secondOpponentDist = opponentDist;
				}

				analysis.opponentCenterOfMass = analysis.opponentCenterOfMass + p.pos;  // Fixed: use + instead of +=
			}
		}

		if (opponentCount > 0) {
			float countFloat = (float)opponentCount;  // Fixed: explicit cast
			analysis.opponentCenterOfMass = analysis.opponentCenterOfMass / countFloat;  // Fixed: explicit division
			analysis.avgOpponentSpeed = totalOpponentSpeed / countFloat;
		}

		return analysis;
	}

	PlayerRole DeterminePlayerRole(const Player& player, const TeamAnalysis& analysis, const GameState& state) {
		float playerDistToBall = (player.pos - state.ball.pos).Length();

		// Factor 1: Distance to ball (40% weight)
		float distanceScore = (playerDistToBall < analysis.teammateDistToBall) ? 0.4f : 0.f;

		// Factor 2: Speed toward ball (30% weight) - Fixed type issues
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();
		Vec teammateToBall = (state.ball.pos - analysis.teammate->pos).Normalized();
		float playerVelToBall = player.vel.Dot(playerToBall);
		float teammateVelToBall = analysis.teammate->vel.Dot(teammateToBall);
		float speedScore = (playerVelToBall > teammateVelToBall) ? 0.3f : 0.f;

		// Factor 3: Boost level consideration (20% weight)
		float boostScore = (player.boost > analysis.teammate->boost + 10.f) ? 0.2f : 0.f;

		// Factor 4: Spawn position advantage (10% weight)
		float spawnScore = CalculateSpawnAdvantage(player, *analysis.teammate, state) * 0.1f;

		float totalScore = distanceScore + speedScore + boostScore + spawnScore;

		return (totalScore >= 0.5f) ? PlayerRole::GOER : PlayerRole::CHEATER;
	}

	float CalculateSpawnAdvantage(const Player& player, const Player& teammate, const GameState& state) {
		// Advantage based on diagonal vs straight kickoff positioning
		float playerAngleToBall = atan2f(player.pos.y - state.ball.pos.y, player.pos.x - state.ball.pos.x);
		float teammateAngleToBall = atan2f(teammate.pos.y - state.ball.pos.y, teammate.pos.x - state.ball.pos.x);

		float angleDiff = fabsf(playerAngleToBall - teammateAngleToBall);

		// Diagonal spawns (corner positions) have advantage for going
		return (angleDiff > (3.14159f / 3.f)) ? 1.f : 0.f;  // Fixed: use explicit float for PI
	}

	float CalculateGoerReward(const Player& player, const TeamAnalysis& analysis, const GameState& state) {
		float playerDistToBall = (player.pos - state.ball.pos).Length();

		// Base reward for being closer than opponents
		float baseReward = (playerDistToBall < analysis.closestOpponentDist) ? goerReward : -goerReward * 0.5f;

		// Speed differential bonus - Fixed type issues
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();
		float playerVelToBall = player.vel.Dot(playerToBall);
		float speedBonus = RS_CLAMP(playerVelToBall / 2300.f, -0.3f, 0.3f); // Max car speed ~2300

		// Boost usage efficiency (penalize waste, reward conservation for crucial moments)
		float boostEfficiency = 0.f;
		if (player.boost > 50.f && playerDistToBall > 1000.f) {
			boostEfficiency = 0.1f; // Good boost management
		}
		else if (player.boost < 20.f && playerDistToBall > 800.f) {
			boostEfficiency = -0.15f; // Poor boost management
		}

		// Angle approach bonus (reward straight-line approaches)
		Vec toBall = (state.ball.pos - player.pos).Normalized();
		Vec velocity = player.vel.Normalized();
		float approachAngle = toBall.Dot(velocity);
		float angleBonus = RS_MAX(0.f, approachAngle) * 0.2f;

		return RS_CLAMP(baseReward + speedBonus + boostEfficiency + angleBonus, -1.5f, 1.5f);
	}

	float CalculateCheaterReward(const Player& player, const TeamAnalysis& analysis, const GameState& state) {
		Vec ownGoal = (player.team == Team::BLUE) ?
			CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;

		// Dynamic ideal position based on game state
		Vec idealPos = CalculateDynamicIdealPosition(player, analysis, state, ownGoal);
		float distToIdeal = (player.pos - idealPos).Length();

		// COMPONENT 1: Dynamic positioning (40% weight)
		float positioningReward = CalculatePositioningReward(player, idealPos, distToIdeal);

		// COMPONENT 2: Strategic boost management (25% weight)
		float boostReward = CalculateStrategicBoostReward(player, state, analysis) * 0.25f;

		// COMPONENT 3: Rotation preparation (20% weight)
		float rotationReward = CalculateRotationPreparation(player, analysis, state) * rotationPrepWeight;

		// COMPONENT 4: Opponent awareness (10% weight)
		float awarenessReward = CalculateOpponentAwareness(player, analysis, state) * 0.1f;

		// COMPONENT 5: Anti-camping with dynamic threshold (5% weight)
		float campingPenalty = CalculateDynamicCampingPenalty(player, ownGoal, state) * 0.05f;

		float totalReward = positioningReward + boostReward + rotationReward + awarenessReward + campingPenalty;

		return RS_CLAMP(totalReward, -0.8f, 0.8f);
	}

	Vec CalculateDynamicIdealPosition(const Player& player, const TeamAnalysis& analysis,
		const GameState& state, const Vec& ownGoal) {
		Vec fieldCenter = Vec(0.f, 0.f, 100.f);  // Fixed: explicit floats

		// Base position: 65% toward center from goal (slightly more aggressive than original)
		Vec centerMultiplied = Vec(fieldCenter.x * 1.3f, fieldCenter.y * 1.3f, fieldCenter.z * 1.3f);  // Fixed: manual multiplication
		Vec baseIdeal = (ownGoal + centerMultiplied) * 0.5f;

		// Adjust based on opponent positioning
		Vec opponentThreatVector = (analysis.opponentCenterOfMass - ownGoal).Normalized();
		opponentThreatVector = Vec(opponentThreatVector.x * 200.f, opponentThreatVector.y * 200.f, opponentThreatVector.z * 200.f);  // Fixed: manual scaling

		// Adjust based on teammate position (create optimal spacing)
		Vec teammateOffset = Vec(0.f, 0.f, 0.f);  // Fixed: explicit floats
		if (analysis.teammate) {
			Vec teammatePos = analysis.teammate->pos;
			float teammateDistFromCenter = (teammatePos - fieldCenter).Length();

			// If teammate is far from center, position closer to support
			if (teammateDistFromCenter > 1500.f) {
				Vec direction = (teammatePos - baseIdeal).Normalized();
				teammateOffset = Vec(direction.x * 300.f, direction.y * 300.f, direction.z * 300.f);  // Fixed: manual scaling
			}
		}

		// Final position with adjustments - Fixed: manual scaling
		Vec threatAdjustment = Vec(opponentThreatVector.x * 0.3f, opponentThreatVector.y * 0.3f, opponentThreatVector.z * 0.3f);
		Vec teammateAdjustment = Vec(teammateOffset.x * 0.2f, teammateOffset.y * 0.2f, teammateOffset.z * 0.2f);
		Vec adjustedIdeal = baseIdeal + threatAdjustment + teammateAdjustment;

		// Clamp to reasonable field boundaries
		adjustedIdeal.x = RS_CLAMP(adjustedIdeal.x, -3000.f, 3000.f);
		adjustedIdeal.y = RS_CLAMP(adjustedIdeal.y, -4000.f, 4000.f);
		adjustedIdeal.z = RS_MAX(adjustedIdeal.z, 17.f); // Ground level

		return adjustedIdeal;
	}

	float CalculatePositioningReward(const Player& player, const Vec& idealPos, float distToIdeal) {
		float optimalRadius = 600.f;
		float acceptableRadius = 1200.f;
		float maxRadius = 2000.f;

		if (distToIdeal <= optimalRadius) {
			// Excellent positioning
			return 0.5f * (1.f - (distToIdeal / optimalRadius));
		}
		else if (distToIdeal <= acceptableRadius) {
			// Good positioning with gradual falloff
			float ratio = (distToIdeal - optimalRadius) / (acceptableRadius - optimalRadius);
			return 0.5f * (1.f - ratio) * 0.7f;
		}
		else if (distToIdeal <= maxRadius) {
			// Poor but acceptable positioning
			float ratio = (distToIdeal - acceptableRadius) / (maxRadius - acceptableRadius);
			return -0.1f * ratio;
		}
		else {
			// Very poor positioning
			return -0.3f;
		}
	}

	float CalculateStrategicBoostReward(const Player& player, const GameState& state, const TeamAnalysis& analysis) {
		// Find strategically important boost pads
		float bestBoostValue = 0.f;

		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
			const Vec& boostPos = CommonValues::BOOST_LOCATIONS[i];

			if (boostPos.z > 72.0f) { // Large boost pad
				float distToBoost = (player.pos - boostPos).Length();

				// Value boost based on multiple factors
				float accessibility = 1.f - RS_CLAMP(distToBoost / 1500.f, 0.f, 1.f);
				float strategicValue = CalculateBoostStrategicValue(boostPos, analysis, state);
				float denyValue = CalculateBoostDenialValue(boostPos, analysis);

				float totalValue = accessibility * (strategicValue + denyValue);
				bestBoostValue = RS_MAX(bestBoostValue, totalValue);
			}
		}

		// Boost level consideration
		float boostLevelFactor = 1.f;
		if (player.boost < 30.f) {
			boostLevelFactor = 1.5f; // More urgent need for boost
		}
		else if (player.boost > 80.f) {
			boostLevelFactor = 0.5f; // Less urgent need
		}

		return bestBoostValue * boostLevelFactor;
	}

	float CalculateBoostStrategicValue(const Vec& boostPos, const TeamAnalysis& analysis, const GameState& state) {
		// Boost pads closer to expected ball trajectory are more valuable
		Vec ballToBoost = (boostPos - state.ball.pos);
		float distToBall = ballToBoost.Length();

		// Corner boosts are generally more valuable for rotations
		bool isCornerBoost = (fabsf(boostPos.x) > 2500.f && fabsf(boostPos.y) > 3500.f);

		float baseValue = isCornerBoost ? 0.8f : 0.6f;
		float proximityValue = 1.f - RS_CLAMP(distToBall / 3000.f, 0.f, 1.f);

		return baseValue * (0.3f + proximityValue * 0.7f);
	}

	float CalculateBoostDenialValue(const Vec& boostPos, const TeamAnalysis& analysis) {
		// Value boost based on opponent accessibility
		float opponentDistToBoost = (analysis.opponentCenterOfMass - boostPos).Length();

		return RS_CLAMP(1.f - (opponentDistToBoost / 2000.f), 0.f, 0.3f);
	}

	float CalculateRotationPreparation(const Player& player, const TeamAnalysis& analysis, const GameState& state) {
		if (!analysis.teammate) return 0.f;

		// Reward positioning that allows quick transition to support teammate
		Vec teammatePos = analysis.teammate->pos;
		Vec supportPosition = CalculateOptimalSupportPosition(teammatePos, state.ball.pos, player.team);

		float distToSupport = (player.pos - supportPosition).Length();
		float supportReadiness = 1.f - RS_CLAMP(distToSupport / 1000.f, 0.f, 1.f);

		// Velocity alignment for quick rotation
		Vec toSupport = (supportPosition - player.pos).Normalized();
		float velocityAlignment = RS_MAX(0.f, player.vel.Normalized().Dot(toSupport));

		return (supportReadiness * 0.7f + velocityAlignment * 0.3f);
	}

	Vec CalculateOptimalSupportPosition(const Vec& teammatePos, const Vec& ballPos, Team team) {
		Vec ownGoal = (team == Team::BLUE) ?
			CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;

		// Position that forms good triangle with teammate and goal
		Vec teammateToGoal = (ownGoal - teammatePos).Normalized();
		Vec perpendicular = Vec(-teammateToGoal.y, teammateToGoal.x, 0.f).Normalized();  // Fixed: explicit float

		// Fixed: manual vector arithmetic
		Vec goalOffset = Vec(teammateToGoal.x * 800.f, teammateToGoal.y * 800.f, teammateToGoal.z * 800.f);
		Vec perpOffset = Vec(perpendicular.x * 600.f, perpendicular.y * 600.f, perpendicular.z * 600.f);
		Vec supportPos = teammatePos + goalOffset + perpOffset;

		return supportPos;
	}

	float CalculateOpponentAwareness(const Player& player, const TeamAnalysis& analysis, const GameState& state) {
		// Reward positioning that maintains good sight lines to opponents
		Vec playerToOpponentCenter = (analysis.opponentCenterOfMass - player.pos).Normalized();
		Vec playerToBall = (state.ball.pos - player.pos).Normalized();

		float awarenessAngle = playerToOpponentCenter.Dot(playerToBall);

		// Good awareness when you can see both ball and opponents
		return RS_CLAMP(awarenessAngle * 0.5f + 0.5f, 0.f, 1.f);
	}

	float CalculateDynamicCampingPenalty(const Player& player, const Vec& ownGoal, const GameState& state) {
		float distToGoal = (player.pos - ownGoal).Length();

		// Dynamic threshold based on game state
		float minDistFromGoal = 800.f; // Base minimum

		// Adjust based on ball position
		float ballDistFromGoal = (state.ball.pos - ownGoal).Length();
		if (ballDistFromGoal < 2000.f) {
			minDistFromGoal *= 0.7f; // Allow closer positioning when ball is near
		}

		if (distToGoal < minDistFromGoal) {
			float penalty = -0.4f * (1.f - (distToGoal / minDistFromGoal));
			return penalty;
		}

		return 0.f;
	}
};

} // namespace RLGC