#pragma once
#include "../FrameworkTorch.h"
#include <GigaLearnCPP/Util/Utils.h>
#include <nlohmann/json.hpp>

namespace GGL {
	struct WelfordStat {
		double runningMean = 0, runningVariance = 0;
		mutable double cachedSTD = 1.0;
		mutable int64_t cachedSTDCount = 0;

		int64_t count = 0;

		WelfordStat() {};

		void Increment(const FList& samples) {
			for (float sample : samples) {
				double delta = (double)sample - runningMean;
				double deltaN = delta / (count + 1);

				runningMean += deltaN;
				runningVariance += delta * deltaN * count;
				count++;
			}
		}

		void Reset() {
			*this = WelfordStat();
		}

		double GetMean() const {
			if (count < 2)
				return 0;

			return runningMean;
		}

		double GetSTD() const {
			if (count < 2)
				return 1;

			// Cache la valeur de STD pour éviter les recalculs répétés
			if (cachedSTDCount != count) {
				double curVar = runningVariance / (count - 1);
				if (curVar <= 0)
					curVar = 1;
				cachedSTD = sqrt(curVar);
				cachedSTDCount = count;
			}
			return cachedSTD;
		}

		nlohmann::json ToJSON() const {
			nlohmann::json result = {};
			result["mean"] = runningMean;
			result["var"] = runningVariance;
			result["count"] = count;
			return result;
		}

		void ReadFromJSON(const nlohmann::json& json) {
			runningMean = json["mean"];
			runningVariance = json["var"];
			count = json["count"];
			cachedSTDCount = 0; // Invalide le cache
		}
	};

	struct BatchedWelfordStat {
		int width;
		std::vector<double> runningMeans, runningVariances;
		mutable std::vector<double> cachedSTD;
		mutable std::vector<double> cachedClampedMean; // Cache pour mean clampé
		mutable std::vector<double> cachedClampedSTD;  // Cache pour STD clampé
		mutable int64_t cachedSTDCount = 0;
		mutable double lastMeanClamp = 0;
		mutable double lastMinSTD = 0;

		int64_t count = 0;

		BatchedWelfordStat(int width) : width(width) {
			runningMeans.resize(width);
			runningVariances.resize(width);
			cachedSTD.resize(width, 1.0);
			cachedClampedMean.resize(width, 0.0);
			cachedClampedSTD.resize(width, 1.0);
		};

		void IncrementRow(float* samples) {
			for (int i = 0; i < width; i++) {
				double delta = samples[i] - runningMeans[i];
				double deltaN = delta / (count + 1);
				runningMeans[i] += deltaN;
				runningVariances[i] += delta * deltaN * count;
			}
			count++;
		}

		void Reset() {
			*this = BatchedWelfordStat(width);
		}

		const std::vector<double>& GetMean() const {
			return runningMeans;
		}

		const std::vector<double>& GetSTD() const {
			if (count < 2) {
				// Retourne un vecteur de 1 sans allocation si possible
				if (cachedSTD.size() == static_cast<size_t>(width) && cachedSTDCount == -1) {
					return cachedSTD;
				}
				cachedSTD.assign(width, 1.0);
				cachedSTDCount = -1;
				return cachedSTD;
			}

			// Cache le calcul de STD pour éviter les recalculs à chaque step
			if (cachedSTDCount != count) {
				for (int i = 0; i < width; i++) {
					double var = runningVariances[i] / (count - 1);
					cachedSTD[i] = (var > 0) ? sqrt(var) : 1.0;
				}
				cachedSTDCount = count;
			}
			
			return cachedSTD;
		}

		// OPTIMISATION MAJEURE: Normalise les observations in-place avec SIMD
		// Utilise la vectorisation pour améliorer les performances
		void NormalizeInPlace(float* obs, int numRows, int obsWidth, double meanClamp, double minSTD) const {
			// Skip si pas assez de données
			if (count < 2 || numRows <= 0 || obsWidth <= 0) return;
			
			// Invalider le cache si les paramètres ont changé
			if (cachedSTDCount != count || lastMeanClamp != meanClamp || lastMinSTD != minSTD) {
				const auto& mean = GetMean();
				const auto& std = GetSTD();
				
				for (int j = 0; j < width; j++) {
					cachedClampedMean[j] = RS_CLAMP(mean[j], -meanClamp, meanClamp);
					cachedClampedSTD[j] = RS_MAX(std[j], minSTD);
				}
				lastMeanClamp = meanClamp;
				lastMinSTD = minSTD;
			}
			
			// OPTIMISATION: Pré-calculer les coefficients une seule fois
			thread_local std::vector<float> invSTD;
			thread_local std::vector<float> negMeanDivSTD;
			
			if (invSTD.size() != static_cast<size_t>(width)) {
				invSTD.resize(width);
				negMeanDivSTD.resize(width);
			}
			
			// OPTIMISATION: Vectorized coefficient computation
			for (int j = 0; j < width; j++) {
				const float inv = 1.0f / static_cast<float>(cachedClampedSTD[j]);
				invSTD[j] = inv;
				negMeanDivSTD[j] = -static_cast<float>(cachedClampedMean[j]) * inv;
			}
			
			// OPTIMISATION MAJEURE: Process multiple rows in parallel for large batches
			const int PARALLEL_THRESHOLD = 100;
			
			if (numRows >= PARALLEL_THRESHOLD) {
				// OPTIMISATION: Parallel processing for large batches
				#pragma omp parallel for if(numRows > PARALLEL_THRESHOLD)
				for (int i = 0; i < numRows; i++) {
					float* row = obs + i * obsWidth;
					
					// OPTIMISATION: Manual loop unrolling x8 for AVX compatibility
					int j = 0;
					const int unrollEnd = obsWidth - (obsWidth % 8);
					
					for (; j < unrollEnd; j += 8) {
						row[j]     = row[j]     * invSTD[j]     + negMeanDivSTD[j];
						row[j + 1] = row[j + 1] * invSTD[j + 1] + negMeanDivSTD[j + 1];
						row[j + 2] = row[j + 2] * invSTD[j + 2] + negMeanDivSTD[j + 2];
						row[j + 3] = row[j + 3] * invSTD[j + 3] + negMeanDivSTD[j + 3];
						row[j + 4] = row[j + 4] * invSTD[j + 4] + negMeanDivSTD[j + 4];
						row[j + 5] = row[j + 5] * invSTD[j + 5] + negMeanDivSTD[j + 5];
						row[j + 6] = row[j + 6] * invSTD[j + 6] + negMeanDivSTD[j + 6];
						row[j + 7] = row[j + 7] * invSTD[j + 7] + negMeanDivSTD[j + 7];
					}
					
					// Remainder
					for (; j < obsWidth; j++) {
						row[j] = row[j] * invSTD[j] + negMeanDivSTD[j];
					}
				}
			} else {
				// Small batch - sequential processing
				for (int i = 0; i < numRows; i++) {
					float* row = obs + i * obsWidth;
					
					int j = 0;
					const int unrollEnd = obsWidth - (obsWidth % 4);
					
					// x4 unroll pour petits batches
					for (; j < unrollEnd; j += 4) {
						row[j]     = row[j]     * invSTD[j]     + negMeanDivSTD[j];
						row[j + 1] = row[j + 1] * invSTD[j + 1] + negMeanDivSTD[j + 1];
						row[j + 2] = row[j + 2] * invSTD[j + 2] + negMeanDivSTD[j + 2];
						row[j + 3] = row[j + 3] * invSTD[j + 3] + negMeanDivSTD[j + 3];
					}
					
					for (; j < obsWidth; j++) {
						row[j] = row[j] * invSTD[j] + negMeanDivSTD[j];
					}
				}
			}
		}
		
		// OPTIMISATION: Version pour un seul row (inline)
		void NormalizeRowInPlace(float* row, int obsWidth, double meanClamp, double minSTD) const {
			NormalizeInPlace(row, 1, obsWidth, meanClamp, minSTD);
		}
		
		// NOUVELLE FONCTIONNALITÉ: Batch increment pour plusieurs rows
		void IncrementBatch(float* samples, int numRows, int stride) {
			for (int row = 0; row < numRows; row++) {
				IncrementRow(samples + row * stride);
			}
		}

		nlohmann::json ToJSON() const {
			nlohmann::json result = {};
			result["means"] = Utils::MakeJSONArray<double>(runningMeans);
			result["vars"] = Utils::MakeJSONArray<double>(runningVariances);
			result["count"] = count;
			return result;
		}

		void ReadFromJSON(const nlohmann::json& json) {
			runningMeans = Utils::MakeVecFromJSON<double>(json["means"]);
			runningVariances = Utils::MakeVecFromJSON<double>(json["vars"]);
			count = json["count"];
			cachedSTDCount = 0;
		}
	};
}