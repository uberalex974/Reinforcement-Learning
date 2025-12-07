#include "StateUtil.h"

// OPTIMISATION MAJEURE: Version inline et sans copie inutile
PhysState RLGC::InvertPhys(const PhysState& physState, bool shouldInvert) {
	if (!shouldInvert) {
		return physState; // Copie directe, pas de calcul
	}
	
	// OPTIMISATION: Éviter les multiplications répétées en utilisant des constantes
	PhysState result;
	
	// Inversion: x=-x, y=-y, z=z
	result.pos.x = -physState.pos.x;
	result.pos.y = -physState.pos.y;
	result.pos.z = physState.pos.z;
	
	result.rotMat.forward.x = -physState.rotMat.forward.x;
	result.rotMat.forward.y = -physState.rotMat.forward.y;
	result.rotMat.forward.z = physState.rotMat.forward.z;
	
	result.rotMat.right.x = -physState.rotMat.right.x;
	result.rotMat.right.y = -physState.rotMat.right.y;
	result.rotMat.right.z = physState.rotMat.right.z;
	
	result.rotMat.up.x = -physState.rotMat.up.x;
	result.rotMat.up.y = -physState.rotMat.up.y;
	result.rotMat.up.z = physState.rotMat.up.z;
	
	result.vel.x = -physState.vel.x;
	result.vel.y = -physState.vel.y;
	result.vel.z = physState.vel.z;
	
	result.angVel.x = -physState.angVel.x;
	result.angVel.y = -physState.angVel.y;
	result.angVel.z = physState.angVel.z;

	return result;
}

PhysState RLGC::MirrorPhysX(const PhysState& physState, bool shouldMirror) {
	if (!shouldMirror) {
		return physState;
	}

	PhysState result = physState;

	result.pos.x = -result.pos.x;

	// Thanks Rolv, JPK, and Kaiyo!
	result.rotMat.forward.x = -result.rotMat.forward.x;
	result.rotMat.right.y = -result.rotMat.right.y;
	result.rotMat.right.z = -result.rotMat.right.z;
	result.rotMat.up.x = -result.rotMat.up.x;

	result.vel.x = -result.vel.x;
	result.angVel.y = -result.angVel.y;
	result.angVel.z = -result.angVel.z;

	return result;
}