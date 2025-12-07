#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>

namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;
		
		// OPTIMISATION MAJEURE: Envoi asynchrone des métriques
		// Évite de bloquer l'entraînement pendant l'envoi
		std::thread sendThread;
		std::queue<Report> pendingReports;
		std::mutex queueMutex;
		std::condition_variable queueCV;
		std::atomic<bool> stopThread{false};

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		// Envoie le rapport de manière asynchrone (non-bloquant)
		void Send(const Report& report);
		
		// Envoie le rapport de manière synchrone (bloquant)
		void SendSync(const Report& report);

		~MetricSender();
		
	private:
		void SendThreadFunc();
	};
}