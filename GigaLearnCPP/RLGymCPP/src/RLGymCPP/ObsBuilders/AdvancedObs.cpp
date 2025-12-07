#include "AdvancedObs.h"
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <RLGymCPP/BasicTypes/Lists.h>

using namespace RLGC;

// OPTIMISATION: Constantes de normalisation pré-calculées
static constexpr float POS_COEF = 1.0f / 2300.0f;
static constexpr float VEL_COEF = 1.0f / 2300.0f;
static constexpr float ANG_VEL_COEF = 1.0f / 5.5f;
static constexpr float BOOST_COEF = 0.01f; // 1/100

// OPTIMISATION MAJEURE: Buffer thread-local pour éviter TOUTE allocation
// Chaque thread a son propre buffer pré-alloué
static thread_local FList g_obsBuffer;
static thread_local bool g_obsBufferInitialized = false;

// OPTIMISATION MAJEURE: Structure alignée pour SIMD
struct alignas(16) Vec4 {
	float x, y, z, w;
	
	Vec4() : x(0), y(0), z(0), w(0) {}
	Vec4(float x_, float y_, float z_, float w_ = 0) : x(x_), y(y_), z(z_), w(w_) {}
	Vec4(const Vec& v, float w_ = 0) : x(v.x), y(v.y), z(v.z), w(w_) {}
	
	// OPTIMISATION: Multiplication scalaire inline
	Vec4 operator*(float s) const { return Vec4(x*s, y*s, z*s, w*s); }
};

// OPTIMISATION MAJEURE: Écriture directe dans le buffer sans push_back
// Utilise un pointeur et avance directement - évite les checks de capacité
class FastObsWriter {
public:
	float* ptr;
	float* const end;
	
	FastObsWriter(float* start, float* bufEnd) : ptr(start), end(bufEnd) {}
	
	// OPTIMISATION: Force inline et évite les branches
	__forceinline void Write(float v) {
		*ptr++ = v;
	}
	
	__forceinline void WriteVec(const Vec& v) {
		ptr[0] = v.x;
		ptr[1] = v.y;
		ptr[2] = v.z;
		ptr += 3;
	}
	
	__forceinline void WriteVecScaled(const Vec& v, float scale) {
		ptr[0] = v.x * scale;
		ptr[1] = v.y * scale;
		ptr[2] = v.z * scale;
		ptr += 3;
	}
	
	// OPTIMISATION: Écriture de 3 floats en une fois (dot product result)
	__forceinline void WriteDotProducts3(
		const Vec& row0, const Vec& row1, const Vec& row2,
		const Vec& v, float scale
	) {
		ptr[0] = (row0.x * v.x + row0.y * v.y + row0.z * v.z) * scale;
		ptr[1] = (row1.x * v.x + row1.y * v.y + row1.z * v.z) * scale;
		ptr[2] = (row2.x * v.x + row2.y * v.y + row2.z * v.z) * scale;
		ptr += 3;
	}
	
	size_t Written() const { return ptr - (end - 256); } // Approximation
};

// OPTIMISATION MAJEURE: Inversion inline sans création d'objet temporaire
// Écrit directement les valeurs inversées
struct InvertedPhys {
	Vec pos, vel, angVel;
	Vec forward, right, up;
	
	// Constructeur depuis Player avec inversion optionnelle
	InvertedPhys(const Player& p, bool inv) {
		if (inv) {
			pos.x = -p.pos.x; pos.y = -p.pos.y; pos.z = p.pos.z;
			vel.x = -p.vel.x; vel.y = -p.vel.y; vel.z = p.vel.z;
			angVel.x = -p.angVel.x; angVel.y = -p.angVel.y; angVel.z = p.angVel.z;
			forward.x = -p.rotMat.forward.x; forward.y = -p.rotMat.forward.y; forward.z = p.rotMat.forward.z;
			right.x = -p.rotMat.right.x; right.y = -p.rotMat.right.y; right.z = p.rotMat.right.z;
			up.x = -p.rotMat.up.x; up.y = -p.rotMat.up.y; up.z = p.rotMat.up.z;
		} else {
			pos = p.pos; vel = p.vel; angVel = p.angVel;
			forward = p.rotMat.forward; right = p.rotMat.right; up = p.rotMat.up;
		}
	}
	
	// Constructeur depuis BallState avec inversion optionnelle
	InvertedPhys(const BallState& b, bool inv) {
		if (inv) {
			pos.x = -b.pos.x; pos.y = -b.pos.y; pos.z = b.pos.z;
			vel.x = -b.vel.x; vel.y = -b.vel.y; vel.z = b.vel.z;
			angVel.x = -b.angVel.x; angVel.y = -b.angVel.y; angVel.z = b.angVel.z;
		} else {
			pos = b.pos; vel = b.vel; angVel = b.angVel;
		}
		// Ball n'a pas de rotation
		forward = right = up = Vec(0, 0, 0);
	}
};

// OPTIMISATION MAJEURE: AddPlayerToObs avec écriture directe dans buffer
static inline void AddPlayerToObsFast(float*& ptr, const Player& player, bool inv, const InvertedPhys& ball) {
	InvertedPhys phys(player, inv);
	
	// Position (3)
	ptr[0] = phys.pos.x * POS_COEF;
	ptr[1] = phys.pos.y * POS_COEF;
	ptr[2] = phys.pos.z * POS_COEF;
	ptr += 3;
	
	// Forward (3)
	ptr[0] = phys.forward.x;
	ptr[1] = phys.forward.y;
	ptr[2] = phys.forward.z;
	ptr += 3;
	
	// Up (3)
	ptr[0] = phys.up.x;
	ptr[1] = phys.up.y;
	ptr[2] = phys.up.z;
	ptr += 3;
	
	// Velocity (3)
	ptr[0] = phys.vel.x * VEL_COEF;
	ptr[1] = phys.vel.y * VEL_COEF;
	ptr[2] = phys.vel.z * VEL_COEF;
	ptr += 3;
	
	// Angular velocity (3)
	ptr[0] = phys.angVel.x * ANG_VEL_COEF;
	ptr[1] = phys.angVel.y * ANG_VEL_COEF;
	ptr[2] = phys.angVel.z * ANG_VEL_COEF;
	ptr += 3;
	
	// Local angular velocity (3) - rotMat.Dot(angVel)
	ptr[0] = (phys.forward.x * phys.angVel.x + phys.forward.y * phys.angVel.y + phys.forward.z * phys.angVel.z) * ANG_VEL_COEF;
	ptr[1] = (phys.right.x * phys.angVel.x + phys.right.y * phys.angVel.y + phys.right.z * phys.angVel.z) * ANG_VEL_COEF;
	ptr[2] = (phys.up.x * phys.angVel.x + phys.up.y * phys.angVel.y + phys.up.z * phys.angVel.z) * ANG_VEL_COEF;
	ptr += 3;
	
	// Local ball pos (3)
	const float relBallX = ball.pos.x - phys.pos.x;
	const float relBallY = ball.pos.y - phys.pos.y;
	const float relBallZ = ball.pos.z - phys.pos.z;
	ptr[0] = (phys.forward.x * relBallX + phys.forward.y * relBallY + phys.forward.z * relBallZ) * POS_COEF;
	ptr[1] = (phys.right.x * relBallX + phys.right.y * relBallY + phys.right.z * relBallZ) * POS_COEF;
	ptr[2] = (phys.up.x * relBallX + phys.up.y * relBallY + phys.up.z * relBallZ) * POS_COEF;
	ptr += 3;
	
	// Local ball vel (3)
	const float relVelX = ball.vel.x - phys.vel.x;
	const float relVelY = ball.vel.y - phys.vel.y;
	const float relVelZ = ball.vel.z - phys.vel.z;
	ptr[0] = (phys.forward.x * relVelX + phys.forward.y * relVelY + phys.forward.z * relVelZ) * VEL_COEF;
	ptr[1] = (phys.right.x * relVelX + phys.right.y * relVelY + phys.right.z * relVelZ) * VEL_COEF;
	ptr[2] = (phys.up.x * relVelX + phys.up.y * relVelY + phys.up.z * relVelZ) * VEL_COEF;
	ptr += 3;
	
	// Player state (5)
	ptr[0] = player.boost * BOOST_COEF;
	ptr[1] = player.isOnGround ? 1.0f : 0.0f;
	ptr[2] = player.HasFlipOrJump() ? 1.0f : 0.0f;
	ptr[3] = player.isDemoed ? 1.0f : 0.0f;
	ptr[4] = player.hasJumped ? 1.0f : 0.0f;
	ptr += 5;
}

// Taille par joueur: 3+3+3+3+3+3+3+3+5 = 29 floats
static constexpr int PLAYER_OBS_SIZE = 29;
// Taille totale max: 9 (ball) + 8 (action) + 34 (boosts) + 29*6 (6 joueurs max) = 225
static constexpr int MAX_OBS_SIZE = 256;

void RLGC::AdvancedObs::AddPlayerToObs(FList& obs, const Player& player, bool inv, const PhysState& ball) {
	// Cette fonction est gardée pour compatibilité mais utilise la version rapide en interne
	const size_t startSize = obs.size();
	obs.resize(startSize + PLAYER_OBS_SIZE);
	float* ptr = obs.data() + startSize;
	
	InvertedPhys ballPhys(BallState{}, false);
	ballPhys.pos = ball.pos;
	ballPhys.vel = ball.vel;
	ballPhys.angVel = ball.angVel;
	
	AddPlayerToObsFast(ptr, player, inv, ballPhys);
}

FList RLGC::AdvancedObs::BuildObs(const Player& player, const GameState& state) {
	// OPTIMISATION MAJEURE: Utiliser le buffer thread-local pré-alloué
	if (!g_obsBufferInitialized) {
		g_obsBuffer.reserve(MAX_OBS_SIZE);
		g_obsBufferInitialized = true;
	}
	g_obsBuffer.clear();
	
	// OPTIMISATION: Pré-allouer la taille exacte et utiliser un pointeur direct
	// Calculer la taille: 9 + 8 + 34 + 29 * numPlayers
	const int numPlayers = static_cast<int>(state.players.size());
	const int totalSize = 9 + 8 + 34 + PLAYER_OBS_SIZE * numPlayers;
	g_obsBuffer.resize(totalSize);
	
	float* ptr = g_obsBuffer.data();
	const bool inv = player.team == Team::ORANGE;
	
	// OPTIMISATION: Créer la balle inversée une seule fois
	InvertedPhys ball(state.ball, inv);
	
	const auto& pads = state.GetBoostPads(inv);
	const auto& padTimers = state.GetBoostPadTimers(inv);
	
	// Ball state (9)
	ptr[0] = ball.pos.x * POS_COEF;
	ptr[1] = ball.pos.y * POS_COEF;
	ptr[2] = ball.pos.z * POS_COEF;
	ptr[3] = ball.vel.x * VEL_COEF;
	ptr[4] = ball.vel.y * VEL_COEF;
	ptr[5] = ball.vel.z * VEL_COEF;
	ptr[6] = ball.angVel.x * ANG_VEL_COEF;
	ptr[7] = ball.angVel.y * ANG_VEL_COEF;
	ptr[8] = ball.angVel.z * ANG_VEL_COEF;
	ptr += 9;
	
	// Previous action (8)
	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++) {
		*ptr++ = player.prevAction[i];
	}
	
	// Boost pads (34) - OPTIMISATION: Loop unrolling x4
	int i = 0;
	for (; i <= CommonValues::BOOST_LOCATIONS_AMOUNT - 4; i += 4) {
		ptr[0] = pads[i]   ? 1.0f : 1.0f / (1.0f + padTimers[i]);
		ptr[1] = pads[i+1] ? 1.0f : 1.0f / (1.0f + padTimers[i+1]);
		ptr[2] = pads[i+2] ? 1.0f : 1.0f / (1.0f + padTimers[i+2]);
		ptr[3] = pads[i+3] ? 1.0f : 1.0f / (1.0f + padTimers[i+3]);
		ptr += 4;
	}
	for (; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		*ptr++ = pads[i] ? 1.0f : 1.0f / (1.0f + padTimers[i]);
	}
	
	// Current player (29)
	AddPlayerToObsFast(ptr, player, inv, ball);
	
	// OPTIMISATION MAJEURE: Une seule boucle avec tri en place
	// Collecter d'abord les indices des coéquipiers puis des adversaires
	for (const auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId) continue;
		if (otherPlayer.team == player.team) {
			AddPlayerToObsFast(ptr, otherPlayer, inv, ball);
		}
	}
	
	for (const auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId) continue;
		if (otherPlayer.team != player.team) {
			AddPlayerToObsFast(ptr, otherPlayer, inv, ball);
		}
	}
	
	// Ajuster la taille finale (au cas où il y a moins de joueurs que prévu)
	const size_t actualSize = ptr - g_obsBuffer.data();
	g_obsBuffer.resize(actualSize);
	
	return g_obsBuffer;
}