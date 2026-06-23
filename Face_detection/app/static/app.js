(() => {
  const API = "";
  
  // Embedded mode handling (hide sidebar & propagate query param)
  const queryParams = new URLSearchParams(window.location.search);
  if (queryParams.get("embed") === "true") {
    const hideSidebar = () => {
      const appEl = document.querySelector(".app");
      if (appEl) {
        appEl.classList.add("embedded");
      }
    };
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", hideSidebar);
    } else {
      hideSidebar();
    }
    
    // Intercept clicks on local links to propagate embed parameter
    document.addEventListener("click", (e) => {
      const link = e.target.closest("a");
      if (link && link.href) {
        try {
          const url = new URL(link.href);
          if (url.origin === window.location.origin && !url.searchParams.has("embed")) {
            url.searchParams.set("embed", "true");
            link.href = url.toString();
          }
        } catch (err) {
          // ignore
        }
      }
    });
  }

  const appConfig = {
    default_threshold: 0.45,
    webcam_interval_ms: 1500,
  };
  let cameraStream = null;
  let recognitionTimer = null;
  let recognitionBusy = false;

  const $ = (id) => document.getElementById(id);

  function setStatus(el, text, kind) {
    if (!el) return;
    el.textContent = text;
    el.classList.remove("good", "bad", "blue", "text-success", "text-danger", "text-warning");
    if (kind === "success") {
      el.classList.add("good", "text-success");
    } else if (kind === "danger") {
      el.classList.add("bad", "text-danger");
    } else if (kind === "warning") {
      el.classList.add("blue", "text-warning");
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function apiJson(url, options = {}) {
    return fetch(url, options).then(async (res) => {
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || data.message || "Request failed");
      return data;
    });
  }

  async function loadAppConfig() {
    try {
      const data = await apiJson(`${API}/api/health`);
      appConfig.default_threshold = data.default_threshold ?? 0.45;
      appConfig.webcam_interval_ms = data.webcam_interval_ms ?? 1500;
      const thresholdInput = $("cameraThreshold");
      if (thresholdInput && !thresholdInput.value) {
        thresholdInput.value = String(appConfig.default_threshold);
      }
    } catch {
      // keep defaults
    }
  }

  function currentDateLabel() {
    const now = new Date();
    return now.toLocaleDateString(undefined, {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  }

  function todayString() {
    return new Date().toLocaleDateString("en-CA");
  }

  function initClock() {
    const liveClock = $("liveClock");
    const liveDate = $("liveDate");
    if (!liveClock || !liveDate) return;
    const tick = () => {
      const now = new Date();
      liveClock.textContent = now.toLocaleTimeString();
      liveDate.textContent = currentDateLabel();
    };
    tick();
    setInterval(tick, 1000);
  }

  async function loadEmployees() {
    const employeesBody = $("employeesBody");
    const employeeCount = $("employeeCount");
    if (!employeesBody) return;

    const data = await apiJson(`${API}/api/employees`);
    const employees = data.employees || [];
    if (employeeCount) employeeCount.textContent = employees.length;

    if (!employees.length) {
      employeesBody.innerHTML = '<tr><td colspan="4" class="muted">No employees registered yet.</td></tr>';
      return;
    }

    employeesBody.innerHTML = employees.map((employee) => `
      <tr>
        <td>${escapeHtml(employee.name)}</td>
        <td>${employee.sample_count ?? 0}</td>
        <td>${escapeHtml(employee.created_at || "")}</td>
        <td class="d-flex gap-1 flex-wrap">
          <button class="btn btn-sm btn-soft edit-employee-btn" data-employee-id="${employee.id}" data-employee-name="${escapeHtml(employee.name)}">Edit</button>
          <button class="btn btn-sm btn-soft view-samples-btn" data-employee-id="${employee.id}" data-employee-name="${escapeHtml(employee.name)}">Samples</button>
          <button class="btn btn-sm btn-accent add-samples-btn" data-employee-id="${employee.id}" data-employee-name="${escapeHtml(employee.name)}">Add photos</button>
          <button class="btn btn-sm btn-danger-soft delete-employee-btn" data-employee-id="${employee.id}" data-employee-name="${escapeHtml(employee.name)}">Delete</button>
        </td>
      </tr>
    `).join("");
  }

  function attendanceQueryParams() {
    const params = new URLSearchParams();
    const search = $("searchAttendance")?.value?.trim();
    const fromDate = $("fromDate")?.value;
    const toDate = $("toDate")?.value;
    if (search) params.set("search", search);
    if (fromDate) params.set("from", fromDate);
    if (toDate) params.set("to", toDate);
    return params;
  }

  async function loadAttendance() {
    const attendanceBody = $("attendanceBody");
    const todayAttendanceCount = $("todayAttendanceCount");
    const attendanceTotal = $("attendanceTotal");
    if (!attendanceBody) return;

    const params = attendanceQueryParams();
    const url = `${window.location.origin}/api/attendance?${params.toString()}`;
    const data = await apiJson(url);
    const records = data.attendance || [];
    if (todayAttendanceCount) {
      todayAttendanceCount.textContent = records.filter((row) => row.attendance_date === todayString()).length;
    }
    if (attendanceTotal) {
      attendanceTotal.textContent = `Showing ${records.length} of ${data.total ?? records.length}`;
    }

    if (!records.length) {
      attendanceBody.innerHTML = '<tr><td colspan="5" class="muted">No attendance records found.</td></tr>';
      return;
    }

    attendanceBody.innerHTML = records.map((row) => `
      <tr>
        <td>${escapeHtml(row.employee_name)}</td>
        <td>${escapeHtml(row.attendance_date)}</td>
        <td>${escapeHtml(row.check_in_time)}</td>
        <td>${escapeHtml(row.check_out_time || "—")}</td>
        <td class="d-flex gap-1 flex-wrap">
          ${row.check_out_time ? "" : `<button class="btn btn-sm btn-soft checkout-btn" data-attendance-id="${row.id}">Check out</button>`}
          <button class="btn btn-sm btn-danger-soft delete-attendance-btn" data-attendance-id="${row.id}" data-employee-name="${escapeHtml(row.employee_name)}" data-attendance-date="${escapeHtml(row.attendance_date)}">Delete</button>
        </td>
      </tr>
    `).join("");
  }

  function updateSummary(data) {
    const faces = Array.isArray(data?.faces) ? data.faces : [];
    const recognized = faces.filter((face) => face.matched).length;
    const unknown = faces.length - recognized;
    const attendanceMarked = faces.filter((face) => face.attendance && face.attendance.marked).length;
    const percent = faces.length ? Math.round((recognized / faces.length) * 100) : 0;

    const liveMatchesCount = $("liveMatchesCount");
    const liveUnknownCount = $("liveUnknownCount");
    const recognizedFacesValue = $("recognizedFacesValue");
    const unknownFacesValue = $("unknownFacesValue");
    const attendanceMarkedValue = $("attendanceMarkedValue");
    const markedCountChip = $("markedCountChip");
    const summaryRing = $("summaryRing");
    const summaryPercent = $("summaryPercent");
    const recognitionStatus = $("recognitionStatus");

    if (liveMatchesCount) liveMatchesCount.textContent = recognized;
    if (liveUnknownCount) liveUnknownCount.textContent = unknown;
    if (recognizedFacesValue) recognizedFacesValue.textContent = recognized;
    if (unknownFacesValue) unknownFacesValue.textContent = unknown;
    if (attendanceMarkedValue) attendanceMarkedValue.textContent = attendanceMarked;
    if (markedCountChip) markedCountChip.textContent = `${attendanceMarked} marked`;
    if (summaryPercent) summaryPercent.textContent = `${percent}%`;
    if (summaryRing) summaryRing.style.setProperty("--ring", `${percent * 3.6}deg`);

    if (recognitionStatus) {
      if (!faces.length) {
        setStatus(recognitionStatus, "Idle", "warning");
      } else if (recognized > 0) {
        setStatus(recognitionStatus, `${recognized} recognized`, "success");
      } else {
        setStatus(recognitionStatus, "Unknown", "danger");
      }
    }
  }

  function formatRecognition(face) {
    const name = face.matched ? face.employee_name : "Unknown";
    const score = typeof face.score === "number" ? face.score.toFixed(4) : "n/a";
    const margin = typeof face.margin === "number" ? ` | Gap: ${face.margin.toFixed(4)}` : "";
    let state = "";
    if (face.state === "marked") state = " | Check-in marked";
    else if (face.state === "checked_out") state = " | Check-out updated";
    else if (face.state === "already_marked") state = " | Already marked today";
    return `${name} | Score: ${score}${margin}${state}`;
  }

  function faceBoxColor(state, matched) {
    if (state === "checked_out") return "#2563eb";
    if (state === "already_marked") return "#f59e0b";
    if (matched) return "#22c55e";
    return "#ef4444";
  }

  async function showEmployeeSamples(employeeId, employeeName) {
    const modal = $("samplesModal");
    const modalBody = $("samplesModalBody");
    const modalTitle = $("samplesModalTitle");
    if (!modal || !modalBody || !modalTitle) return;

    modalTitle.textContent = `Samples — ${employeeName}`;
    modalBody.innerHTML = '<div class="muted">Loading...</div>';
    modal.classList.remove("d-none");

    try {
      const data = await apiJson(`${API}/api/employees/${employeeId}/samples`);
      if (!data.samples?.length) {
        modalBody.innerHTML = '<div class="muted">No samples found.</div>';
        return;
      }
      modalBody.innerHTML = `<div class="row g-2">${data.samples.map((sample) => `
        <div class="col-4">
          <img src="${sample.image_url}?t=${Date.now()}" alt="Sample ${sample.id}" class="w-100 rounded border">
        </div>
      `).join("")}</div>`;
    } catch (error) {
      modalBody.innerHTML = `<div class="text-danger">${escapeHtml(error.message)}</div>`;
    }
  }

  function initEmployeesPage() {
    const registerForm = $("registerForm");
    const registerStatus = $("registerStatus");
    const resetRegister = $("resetRegister");
    const employeesBody = $("employeesBody");
    const refreshEmployees = $("refreshEmployees");
    const addSamplesInput = $("addSamplesInput");
    const samplesModal = $("samplesModal");
    const closeSamplesModal = $("closeSamplesModal");
    let pendingAddSamplesEmployeeId = null;
    const employeeCamera = $("employeeCamera");
    const employeeCameraStatus = $("employeeCameraStatus");
    const startEmployeeCamera = $("startEmployeeCamera");
    const stopEmployeeCamera = $("stopEmployeeCamera");
    const captureEmployeeSample = $("captureEmployeeSample");
    const faceImagesInput = $("faceImages");
    let employeeCameraStream = null;
    let capturedEmployeeFiles = [];

    const syncEmployeeFiles = () => {
      if (!faceImagesInput) return;
      const dataTransfer = new DataTransfer();
      capturedEmployeeFiles.forEach((file) => dataTransfer.items.add(file));
      faceImagesInput.files = dataTransfer.files;
    };

    const setEmployeeCameraStatus = (message, kind) => {
      if (employeeCameraStatus) setStatus(employeeCameraStatus, message, kind);
    };

    const stopEmployeeCameraStream = () => {
      if (employeeCameraStream) {
        employeeCameraStream.getTracks().forEach((track) => track.stop());
        employeeCameraStream = null;
      }
      if (employeeCamera) employeeCamera.srcObject = null;
      setEmployeeCameraStatus(capturedEmployeeFiles.length ? "" + capturedEmployeeFiles.length + " captured sample(s) ready." : "Camera stopped.", capturedEmployeeFiles.length ? "success" : "warning");
    };

    const startEmployeeCameraStream = async () => {
      employeeCameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      if (employeeCamera) {
        employeeCamera.srcObject = employeeCameraStream;
        await employeeCamera.play();
      }
      setEmployeeCameraStatus("Camera started. Capture 3 to 5 face images.", "success");
    };

    const captureEmployeeFrame = async () => {
      if (!employeeCameraStream) {
        await startEmployeeCameraStream();
      }
      if (!employeeCamera || !employeeCameraStream) return;
      const canvas = document.createElement("canvas");
      canvas.width = employeeCamera.videoWidth || 1280;
      canvas.height = employeeCamera.videoHeight || 720;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(employeeCamera, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
      if (!blob) {
        setEmployeeCameraStatus("Could not capture the frame.", "danger");
        return;
      }
      const file = new File([blob], "face-sample-" + (capturedEmployeeFiles.length + 1) + ".jpg", { type: "image/jpeg" });
      capturedEmployeeFiles = [...capturedEmployeeFiles, file];
      syncEmployeeFiles();
      setEmployeeCameraStatus(capturedEmployeeFiles.length + " sample(s) attached to the form.", "success");
    };

    if (registerForm) {      registerForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const name = $("employeeName").value.trim();
        const files = $("faceImages").files;

        if (!name) {
          setStatus(registerStatus, "Enter an employee name.", "warning");
          return;
        }
        if (files.length < 3 || files.length > 5) {
          setStatus(registerStatus, "Upload between 3 and 5 images.", "warning");
          return;
        }

        const formData = new FormData();
        formData.append("name", name);
        Array.from(files).forEach((file) => formData.append("images", file));

        try {
          const data = await apiJson(`${API}/api/employees/register`, {
            method: "POST",
            body: formData,
          });
          setStatus(registerStatus, `${data.message}: ${data.employee.name} (${data.employee.sample_count} images)`, "success");
          registerForm.reset();
          capturedEmployeeFiles = [];
          syncEmployeeFiles();
          stopEmployeeCameraStream();
          await loadEmployees();
        } catch (error) {
          setStatus(registerStatus, error.message, "danger");
        }
      });
    }

    resetRegister?.addEventListener("click", () => {
      registerForm?.reset();
      capturedEmployeeFiles = [];
      syncEmployeeFiles();
      stopEmployeeCameraStream();
      setStatus(registerStatus, "Ready.");
      setEmployeeCameraStatus("Use the camera to capture 3 to 5 images before registration.");
    });

    startEmployeeCamera?.addEventListener("click", async () => {
      try {
        await startEmployeeCameraStream();
      } catch (error) {
        setEmployeeCameraStatus(error.message, "danger");
      }
    });

    stopEmployeeCamera?.addEventListener("click", stopEmployeeCameraStream);

    captureEmployeeSample?.addEventListener("click", async () => {
      try {
        await captureEmployeeFrame();
      } catch (error) {
        setEmployeeCameraStatus(error.message, "danger");
      }
    });

    refreshEmployees?.addEventListener("click", () => {
      loadEmployees().catch((error) => setStatus(registerStatus, error.message, "danger"));
    });

    closeSamplesModal?.addEventListener("click", () => {
      samplesModal?.classList.add("d-none");
    });

    addSamplesInput?.addEventListener("change", async (event) => {
      const files = event.target.files;
      if (!pendingAddSamplesEmployeeId || !files?.length) return;
      if (files.length < 1 || files.length > 3) {
        setStatus(registerStatus, "Select 1 to 3 images.", "warning");
        return;
      }
      const formData = new FormData();
      Array.from(files).forEach((file) => formData.append("images", file));
      try {
        const data = await apiJson(`${API}/api/employees/${pendingAddSamplesEmployeeId}/samples`, {
          method: "POST",
          body: formData,
        });
        setStatus(registerStatus, `${data.message}. Total samples: ${data.employee.sample_count}`, "success");
        await loadEmployees();
      } catch (error) {
        setStatus(registerStatus, error.message, "danger");
      } finally {
        pendingAddSamplesEmployeeId = null;
        addSamplesInput.value = "";
      }
    });

    employeesBody?.addEventListener("click", async (event) => {
      const editBtn = event.target.closest(".edit-employee-btn");
      const samplesBtn = event.target.closest(".view-samples-btn");
      const addBtn = event.target.closest(".add-samples-btn");
      const deleteBtn = event.target.closest(".delete-employee-btn");

      if (editBtn) {
        const employeeId = editBtn.dataset.employeeId;
        const currentName = editBtn.dataset.employeeName;
        const newName = window.prompt("Employee name", currentName);
        if (!newName || newName.trim() === currentName) return;
        try {
          const formData = new FormData();
          formData.append("name", newName.trim());
          await apiJson(`${API}/api/employees/${employeeId}`, { method: "PUT", body: formData });
          await loadEmployees();
          setStatus(registerStatus, "Employee name updated.", "success");
        } catch (error) {
          setStatus(registerStatus, error.message, "danger");
        }
        return;
      }

      if (samplesBtn) {
        await showEmployeeSamples(samplesBtn.dataset.employeeId, samplesBtn.dataset.employeeName);
        return;
      }

      if (addBtn) {
        pendingAddSamplesEmployeeId = addBtn.dataset.employeeId;
        addSamplesInput?.click();
        return;
      }

      if (deleteBtn) {
        const employeeId = deleteBtn.dataset.employeeId;
        const employeeName = deleteBtn.dataset.employeeName;
        const confirmed = window.confirm(`Delete ${employeeName}? This will remove the employee, attendance records, and uploaded samples.`);
        if (!confirmed) return;
        try {
          await apiJson(`${API}/api/employees/${encodeURIComponent(employeeId)}`, { method: "DELETE" });
          await loadEmployees();
          setStatus(registerStatus, `Deleted ${employeeName}.`, "success");
        } catch (error) {
          setStatus(registerStatus, error.message, "danger");
        }
      }
    });

    loadEmployees().catch((error) => {
      if (registerStatus) setStatus(registerStatus, error.message, "danger");
    });
  }

  function initAttendancePage() {
    const searchBtn = $("searchBtn");
    const refreshAttendance = $("refreshAttendance");
    const exportAttendance = $("exportAttendance");
    const searchAttendance = $("searchAttendance");
    const attendanceBody = $("attendanceBody");
    const manualForm = $("manualAttendanceForm");
    const manualStatus = $("manualAttendanceStatus");
    if (!searchBtn && !refreshAttendance && !searchAttendance) return;

    const load = () => loadAttendance();
    searchBtn?.addEventListener("click", load);
    refreshAttendance?.addEventListener("click", load);
    searchAttendance?.addEventListener("input", (event) => {
      if (!event.target.value.trim()) loadAttendance();
    });

    // Default date filters to today so records are visible on page open
    const todayISO = todayString();
    if ($('fromDate') && !$('fromDate').value) $('fromDate').value = todayISO;
    if ($('toDate') && !$('toDate').value) $('toDate').value = todayISO;

    exportAttendance?.addEventListener("click", () => {
      const params = attendanceQueryParams();
      window.location.href = `${window.location.origin}/api/attendance/export.csv?${params.toString()}`;
    });

    manualForm?.addEventListener("submit", async (event) => {
      event.preventDefault();
      const formData = new FormData(manualForm);
      try {
        const data = await apiJson(`${API}/api/attendance/manual`, { method: "POST", body: formData });
        setStatus(manualStatus, data.message, "success");
        manualForm.reset();
        await loadAttendance();
      } catch (error) {
        setStatus(manualStatus, error.message, "danger");
      }
    });

    attendanceBody?.addEventListener("click", async (event) => {
      const checkoutBtn = event.target.closest(".checkout-btn");
      const deleteBtn = event.target.closest(".delete-attendance-btn");

      if (checkoutBtn) {
        try {
          await apiJson(`${API}/api/attendance/${checkoutBtn.dataset.attendanceId}/checkout`, { method: "POST" });
          await loadAttendance();
        } catch (error) {
          alert(error.message);
        }
        return;
      }

      if (deleteBtn) {
        const attendanceId = deleteBtn.dataset.attendanceId;
        const employeeName = deleteBtn.dataset.employeeName;
        const attendanceDate = deleteBtn.dataset.attendanceDate;
        const confirmed = window.confirm(`Delete attendance for ${employeeName} on ${attendanceDate}?`);
        if (!confirmed) return;
        try {
          await apiJson(`${API}/api/attendance/${encodeURIComponent(attendanceId)}`, { method: "DELETE" });
          await loadAttendance();
        } catch (error) {
          alert(error.message);
        }
      }
    });

    Promise.all([
      apiJson(`${API}/api/employees`),
      loadAttendance(),
    ]).then(([employeesData]) => {
      const select = $("manualEmployeeId");
      if (!select) return;
      select.innerHTML = (employeesData.employees || []).map((employee) =>
        `<option value="${employee.id}">${escapeHtml(employee.name)}</option>`
      ).join("") || '<option value="">No employees</option>';
    }).catch(() => { });
  }

  function resizeOverlay() {
    const webcam = $("webcam");
    const overlay = $("overlay");
    if (!webcam || !overlay) return;
    overlay.width = webcam.videoWidth || 1280;
    overlay.height = webcam.videoHeight || 720;
  }

  function clearOverlay() {
    const overlay = $("overlay");
    const ctx = overlay?.getContext("2d");
    if (!overlay || !ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function drawFaces(data) {
    const webcam = $("webcam");
    const overlay = $("overlay");
    if (!webcam || !overlay) return;

    const ctx = overlay.getContext("2d");
    if (!ctx) return;

    resizeOverlay();
    clearOverlay();
    ctx.lineWidth = 3;
    ctx.font = "16px Segoe UI, sans-serif";
    ctx.textBaseline = "top";

    if (!data || !Array.isArray(data.faces)) return;

    for (const face of data.faces) {
      if (!Array.isArray(face.box) || face.box.length < 4) continue;
      const [x1, y1, x2, y2] = face.box;
      const label = `${face.matched ? face.employee_name : "Unknown"} ${typeof face.score === "number" ? face.score.toFixed(2) : ""}`.trim();
      const color = faceBoxColor(face.state, face.matched);
      ctx.strokeStyle = color;
      ctx.fillStyle = color + "22";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const labelWidth = ctx.measureText(label).width + 14;
      const labelY = Math.max(0, y1 - 26);
      ctx.fillStyle = color;
      ctx.fillRect(x1, labelY, labelWidth, 22);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, x1 + 7, labelY + 3);
    }
  }

  function initWebcamPage() {
    const webcam = $("webcam");
    const open = $("startCamera");
    const stop = $("stopCamera");
    const cameraStatus = $("cameraStatus");
    const attendanceToast = $("attendanceToast");
    const recognitionStatus = $("recognitionStatus");
    const summaryRing = $("summaryRing");
    if (!webcam || !open || !stop || !cameraStatus || !attendanceToast || !recognitionStatus || !summaryRing) return;

    const scratchCanvas = document.createElement("canvas");
    const scratchCtx = scratchCanvas.getContext("2d");

    async function captureFrameBlob() {
      if (!cameraStream) await openCamera();
      const width = webcam.videoWidth || 1280;
      const height = webcam.videoHeight || 720;
      scratchCanvas.width = width;
      scratchCanvas.height = height;
      scratchCtx.drawImage(webcam, 0, 0, width, height);
      return await new Promise((resolve) => scratchCanvas.toBlob(resolve, "image/jpeg", 0.85));
    }

    async function openCamera() {
      if (cameraStream) return;
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      webcam.srcObject = cameraStream;
      webcam.onloadedmetadata = () => resizeOverlay();
      setStatus(cameraStatus, "Webcam opened. Recognition running.", "success");
      setStatus(recognitionStatus, "Scanning", "warning");
      startRecognitionLoop();
    }

    function stopCamera() {
      stopRecognitionLoop();
      if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
      }
      webcam.srcObject = null;
      clearOverlay();
      setStatus(cameraStatus, "Webcam stopped.", "warning");
      setStatus(recognitionStatus, "Idle", "warning");
    }

    function startRecognitionLoop() {
      stopRecognitionLoop();
      recognitionTimer = setInterval(recognizeFrame, appConfig.webcam_interval_ms);
      recognizeFrame();
    }

    function stopRecognitionLoop() {
      if (recognitionTimer) {
        clearInterval(recognitionTimer);
        recognitionTimer = null;
      }
      recognitionBusy = false;
    }

    async function recognizeFrame() {
      if (!cameraStream || recognitionBusy) return;
      recognitionBusy = true;
      try {
        const blob = await captureFrameBlob();
        if (!blob) return;
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");
        const data = await apiJson(`${API}/api/recognize-frame?threshold=${appConfig.default_threshold}`, {
          method: "POST",
          body: formData,
        });
        drawFaces(data);
        updateSummary(data);
        const faces = Array.isArray(data.faces) ? data.faces : [];
        if (faces.length) {
          const lines = faces.map((face, index) => `Face ${index + 1}: ${formatRecognition(face)}`);
          setStatus(cameraStatus, lines.join("\n"), data.status ? "success" : "warning");
          const markedNames = faces.filter((face) => face.state === "marked" || face.state === "marked_in").map((face) => face.employee_name);
          const checkoutNames = faces.filter((face) => face.state === "checked_out" || face.state === "marked_out").map((face) => face.employee_name);
          const repeatMarkedNames = faces.filter((face) => face.state === "already_marked").map((face) => face.employee_name);
          if (markedNames.length) {
            attendanceToast.textContent = `Check-in marked for: ${markedNames.join(", ")}`;
          } else if (checkoutNames.length) {
            attendanceToast.textContent = `Check-out updated for: ${checkoutNames.join(", ")}`;
          } else if (repeatMarkedNames.length) {
            attendanceToast.textContent = `Already marked today: ${repeatMarkedNames.join(", ")}`;
          } else if (faces.some((face) => face.matched)) {
            attendanceToast.textContent = "Recognized employee, no new attendance action.";
          } else {
            attendanceToast.textContent = "No registered employee matched in this frame.";
          }
        } else {
          setStatus(cameraStatus, "No face detected.", "warning");
          attendanceToast.textContent = "Waiting for a recognized employee.";
          updateSummary({ faces: [] });
        }
      } catch (error) {
        setStatus(cameraStatus, error.message, "danger");
      } finally {
        recognitionBusy = false;
      }
    }

    open.addEventListener("click", async () => {
      try {
        await openCamera();
      } catch (error) {
        setStatus(cameraStatus, error.message || "Could not open webcam.", "danger");
      }
    });
    stop.addEventListener("click", stopCamera);
    window.addEventListener("resize", resizeOverlay);
  }

  function initDashboardPage() {
    const dashboardSummary = $("dashboardSummary");
    if (!dashboardSummary) return;
    Promise.all([
      apiJson(`${API}/api/employees`),
      apiJson(`${window.location.origin}/api/attendance`),
    ]).then(([employeesData, attendanceData]) => {
      const employees = employeesData.employees || [];
      const records = attendanceData.attendance || [];
      const today = todayString();
      const todayRecords = records.filter((row) => row.attendance_date === today);
      $("employeeCount") && ($("employeeCount").textContent = employees.length);
      $("todayAttendanceCount") && ($("todayAttendanceCount").textContent = todayRecords.length);
      $("dashboardAttendanceCount") && ($("dashboardAttendanceCount").textContent = attendanceData.total ?? records.length);
      $("dashboardLatestCount") && ($("dashboardLatestCount").textContent = records.slice(0, 3).length);
    }).catch(() => { });
  }

  function initHealth() {
    const apiStatus = $("apiStatus");
    if (!apiStatus) return;
    apiJson(`${API}/api/health`)
      .then(() => setStatus(apiStatus, "Online", "success"))
      .catch(() => setStatus(apiStatus, "Offline", "danger"));
  }

  function initCamerasPage() {
    const cameraForm = $("cameraForm");
    const cameraFormStatus = $("cameraFormStatus");
    const cameraCards = $("cameraCards");
    const refreshCameras = $("refreshCameras");
    const testCameraBtn = $("testCameraBtn");
    if (!cameraForm && !cameraCards) return;

    function runtimeLabel(runtime) {
      const status = runtime?.status || "stopped";
      if (status === "running") return { text: "Running", kind: "success" };
      if (status === "reconnecting") return { text: "Reconnecting", kind: "warning" };
      if (status === "error") return { text: "Error", kind: "danger" };
      return { text: "Stopped", kind: "warning" };
    }

    function renderCameraCards(cameras) {
      if (!cameraCards) return;
      if (!cameras.length) {
        cameraCards.innerHTML = '<div class="muted">No CCTV cameras configured yet.</div>';
        return;
      }

      cameraCards.innerHTML = cameras.map((camera) => {
        const runtime = camera.runtime || {};
        const badge = runtimeLabel(runtime);
        const faces = runtime.latest_result?.faces || [];
        const faceLines = faces.length
          ? faces.map((face, index) => `<div class="small-note">Face ${index + 1}: ${escapeHtml(face.employee_name || "Unknown")} (${typeof face.score === "number" ? face.score.toFixed(3) : "n/a"})</div>`).join("")
          : '<div class="small-note">No faces in the latest frame.</div>';
        const previewUrl = `${API}/api/cameras/${camera.id}/preview.jpg?t=${Date.now()}`;

        return `
          <div class="panel-soft p-3" data-camera-id="${camera.id}">
            <div class="d-flex justify-content-between align-items-start flex-wrap gap--2 mb-3">
              <div>
                <div class="fw-bold">${escapeHtml(camera.name)}</div>
                <div class="small-note">${escapeHtml(camera.source_type.toUpperCase())}: ${escapeHtml(camera.source_url)}</div>
                <div class="small-note">Threshold ${camera.threshold} · every ${camera.interval_sec}s</div>
              </div>
              <span class="chip ${badge.kind === "success" ? "good" : badge.kind === "danger" ? "bad" : ""}">${badge.text}</span>
            </div>
            <div class="row g-3">
              <div class="col-md-7">
                <div class="video-wrap" style="aspect-ratio: 16 / 10;">
                  <img src="${previewUrl}" alt="${escapeHtml(camera.name)} preview" class="w-100 h-100 object-fit-cover" onerror="this.style.display='none'">
                </div>
              </div>
              <div class="col-md-5">
                ${faceLines}
                ${runtime.last_error ? `<div class="small-note text-danger mt-2">${escapeHtml(runtime.last_error)}</div>` : ""}
                <div class="d-flex gap-2 flex-wrap mt-3">
                  <button class="btn btn-sm btn-accent start-camera-btn" data-camera-id="${camera.id}">Start</button>
                  <button class="btn btn-sm btn-soft stop-camera-btn" data-camera-id="${camera.id}">Stop</button>
                  <button class="btn btn-sm btn-soft edit-camera-btn" data-camera-id="${camera.id}" data-camera-name="${escapeHtml(camera.name)}" data-camera-source="${escapeHtml(camera.source_url)}" data-camera-type="${camera.source_type}" data-camera-threshold="${camera.threshold}" data-camera-interval="${camera.interval_sec}">Edit</button>
                  <button class="btn btn-sm btn-danger-soft delete-camera-btn" data-camera-id="${camera.id}" data-camera-name="${escapeHtml(camera.name)}">Delete</button>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join("");
    }

    async function loadCameras() {
      const data = await apiJson(`${API}/api/cameras`);
      renderCameraCards(data.cameras || []);
    }

    testCameraBtn?.addEventListener("click", async () => {
      const formData = new FormData();
      formData.append("source_url", $("cameraSource").value.trim());
      formData.append("source_type", $("cameraType").value);
      try {
        const data = await apiJson(`${API}/api/cameras/test`, { method: "POST", body: formData });
        setStatus(cameraFormStatus, data.message, data.ok ? "success" : "danger");
      } catch (error) {
        setStatus(cameraFormStatus, error.message, "danger");
      }
    });

    if (cameraForm) {
      cameraForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append("name", $("cameraName").value.trim());
        formData.append("source_url", $("cameraSource").value.trim());
        formData.append("source_type", $("cameraType").value);
        formData.append("threshold", $("cameraThreshold").value || String(appConfig.default_threshold));
        formData.append("interval_sec", $("cameraInterval").value || "1.5");
        formData.append("enabled", "true");

        try {
          const data = await apiJson(`${API}/api/cameras`, { method: "POST", body: formData });
          setStatus(cameraFormStatus, `${data.message}: ${data.camera.name}`, "success");
          cameraForm.reset();
          $("cameraThreshold").value = String(appConfig.default_threshold);
          $("cameraInterval").value = "1.5";
          await loadCameras();
        } catch (error) {
          setStatus(cameraFormStatus, error.message, "danger");
        }
      });
    }

    refreshCameras?.addEventListener("click", () => {
      loadCameras().catch((error) => setStatus(cameraFormStatus, error.message, "danger"));
    });

    cameraCards?.addEventListener("click", async (event) => {
      const startBtn = event.target.closest(".start-camera-btn");
      const stopBtn = event.target.closest(".stop-camera-btn");
      const editBtn = event.target.closest(".edit-camera-btn");
      const deleteBtn = event.target.closest(".delete-camera-btn");

      try {
        if (startBtn) {
          await apiJson(`${API}/api/cameras/${encodeURIComponent(startBtn.dataset.cameraId)}/start`, { method: "POST" });
          await loadCameras();
        } else if (stopBtn) {
          await apiJson(`${API}/api/cameras/${encodeURIComponent(stopBtn.dataset.cameraId)}/stop`, { method: "POST" });
          await loadCameras();
        } else if (editBtn) {
          const cameraId = editBtn.dataset.cameraId;
          const name = window.prompt("Camera name", editBtn.dataset.cameraName);
          if (!name) return;
          const threshold = window.prompt("Threshold (0.30 - 0.90)", editBtn.dataset.cameraThreshold);
          const interval = window.prompt("Interval seconds", editBtn.dataset.cameraInterval);
          await apiJson(`${API}/api/cameras/${cameraId}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              name: name.trim(),
              threshold: threshold ? Number(threshold) : undefined,
              interval_sec: interval ? Number(interval) : undefined,
            }),
          });
          await loadCameras();
        } else if (deleteBtn) {
          const confirmed = window.confirm(`Delete camera ${deleteBtn.dataset.cameraName}?`);
          if (!confirmed) return;
          await apiJson(`${API}/api/cameras/${encodeURIComponent(deleteBtn.dataset.cameraId)}`, { method: "DELETE" });
          await loadCameras();
        }
      } catch (error) {
        setStatus(cameraFormStatus, error.message, "danger");
      }
    });

    loadCameras().catch((error) => setStatus(cameraFormStatus, error.message, "danger"));
    setInterval(() => {
      loadCameras().catch(() => { });
    }, 4000);
  }

  document.addEventListener("DOMContentLoaded", async () => {
    await loadAppConfig();
    initClock();
    initHealth();
    initEmployeesPage();
    initAttendancePage();
    initWebcamPage();
    initCamerasPage();
    initDashboardPage();
  });
})();
