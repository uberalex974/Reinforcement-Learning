#include "MetricSender.h"

#include "Timer.h"

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	// OPTIMISATION: Démarrer le thread d'envoi asynchrone
	sendThread = std::thread(&MetricSender::SendThreadFunc, this);

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::SendThreadFunc() {
	// CORRECTION CRITIQUE: On ne doit PAS garder le GIL pendant l'attente
	// car cela bloquerait tout autre code Python dans l'application
	
	while (true) {
		Report reportToSend;
		bool hasReport = false;
		
		// Attendre SANS le GIL
		{
			std::unique_lock<std::mutex> lock(queueMutex);
			
			// Attendre qu'un rapport soit disponible ou que le thread doive s'arrêter
			queueCV.wait(lock, [this]() {
				return !pendingReports.empty() || stopThread.load();
			});
			
			if (stopThread.load() && pendingReports.empty()) {
				break;
			}
			
			if (!pendingReports.empty()) {
				reportToSend = std::move(pendingReports.front());
				pendingReports.pop();
				hasReport = true;
			}
		}
		
		if (hasReport) {
			// Acquérir le GIL SEULEMENT pour l'appel Python
			py::gil_scoped_acquire acquire;
			
			// Envoyer le rapport (avec GIL acquis)
			py::dict reportDict = {};
			for (auto& pair : reportToSend.data)
				reportDict[pair.first.c_str()] = pair.second;

			try {
				pyMod.attr("add_metrics")(reportDict);
			} catch (std::exception& e) {
				RG_LOG("MetricSender: Failed to add metrics (async), exception: " << e.what());
			}
			// Le GIL est automatiquement relâché ici à la fin du scope
		}
	}
}

void GGL::MetricSender::Send(const Report& report) {
	// OPTIMISATION: Ajouter le rapport à la queue et retourner immédiatement
	{
		std::lock_guard<std::mutex> lock(queueMutex);
		pendingReports.push(report);
	}
	queueCV.notify_one();
}

void GGL::MetricSender::SendSync(const Report& report) {
	// Version synchrone pour les cas où on doit attendre
	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	// Signaler au thread de s'arrêter et attendre qu'il finisse
	stopThread.store(true);
	queueCV.notify_one();
	
	if (sendThread.joinable()) {
		sendThread.join();
	}
}