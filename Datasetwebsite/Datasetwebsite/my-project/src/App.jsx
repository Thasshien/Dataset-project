import React, { useState, useEffect } from 'react';
import { Upload, FileText, BarChart3, TrendingUp, AlertCircle, CheckCircle, Clock, BookOpen, User, LogOut, ChevronRight, Download, Eye, Target, Zap, Settings, Menu, X, GraduationCap, Award, Plus, Trash2, Calendar, Send, EyeOff, Users } from 'lucide-react';
// import Aurora from './Aurora';

// ==================== DOTGRID COMPONENT ====================
import DotGrid from './components/Dotgrid';


const LoginPage = ({ onLogin }) => {
  const [userType, setUserType] = useState(null);
  const [credentials, setCredentials] = useState({ email: '', password: '' });
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = () => {
    if (credentials.email && credentials.password) {
      const user = {
        email: credentials.email,
        name: userType === 'teacher' ? 'Dr. Sarah Chen' : 'Alex Morgan',
        id: credentials.password
      };
      onLogin(user, userType);
    }
  };

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-gray-70 via-indigo-50 to-purple-50">
      <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <DotGrid
          dotSize={8}
          gap={20}
          baseColor="#5227FF"
          activeColor="#5227FF"
          proximity={150}
          shockRadius={300}
          shockStrength={8}
          resistance={600}
          returnDuration={2}
          opacity={0.3}
        />
      </div>

      <div className="relative z-10 min-h-screen flex items-center justify-center p-4">
        <div className="bg-white/90 backdrop-blur-xl rounded-3xl shadow-2xl w-full max-w-md border border-white/20 overflow-hidden">
          {!userType ? (
            <>
              <div className="bg-gradient-to-r from-blue-700 to-indigo-900 p-8 text-white">
                <div className="flex items-center justify-center mb-4">
                  <div className="w-16 h-16 bg-white/20 rounded-2xl flex items-center justify-center">
                    <GraduationCap className="w-10 h-10" />

                  </div>
                </div>
                <h1 className="text-3xl font-bold text-center">EduAssign AI</h1>
                <p className="text-center text-blue-100 mt-2">Smart Assignment Management</p>
              </div>

              <div className="p-8">
                <h2 className="text-2xl font-semibold text-gray-800 mb-6 text-center">Select Your Role</h2>
                <div className="space-y-4">
                  <button
                    onClick={() => setUserType('teacher')}
                    className="w-full bg-gradient-to-r from-blue-500 to-blue-600 text-white py-4 rounded-xl font-semibold hover:from-blue-600 hover:to-blue-700 transition-all transform hover:scale-105 flex items-center justify-center gap-3 shadow-lg"
                  >
                    <Users className="w-6 h-6" />
                    Login as Teacher
                  </button>
                  <button
                    onClick={() => setUserType('student')}
                    className="w-full bg-gradient-to-r  from-blue-500 to-blue-600 text-white py-4 rounded-xl font-semibold  hover:from-blue-600 hover:to-blue-700 transition-all transform hover:scale-105 flex items-center justify-center gap-3 shadow-lg"
                  >
                    <BookOpen className="w-6 h-6" />
                    Login as Student
                  </button>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6 text-white">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl font-semibold">
                    {userType === 'teacher' ? 'Teacher' : 'Student'} Login
                  </h2>
                  <button
                    onClick={() => {
                      setUserType(null);
                      setCredentials({ email: '', password: '' });
                    }}
                    className="text-sm bg-white/20 hover:bg-white/30 px-3 py-1 rounded-lg"
                  >
                    Change Role
                  </button>
                </div>
              </div>

              <div className="p-8">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
                    <input
                      type="email"
                      value={credentials.email}
                      onChange={(e) => setCredentials({ ...credentials, email: e.target.value })}
                      placeholder={userType === 'teacher' ? 'teacher@school.edu' : 'student@school.edu'}
                      className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-indigo-500 bg-gray-50 focus:bg-white transition-all"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Password</label>
                    <div className="relative">
                      <input
                        type={showPassword ? 'text' : 'password'}
                        value={credentials.password}
                        onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
                        onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
                        placeholder="Enter your password"
                        className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:outline-none focus:border-indigo-500 bg-gray-50 focus:bg-white transition-all pr-12"
                      />
                      <button
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500 hover:text-gray-700"
                      >
                        {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                      </button>
                    </div>
                  </div>

                  <button
                    onClick={handleLogin}
                    className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3.5 rounded-xl font-semibold hover:shadow-lg hover:scale-[1.02] transition-all"
                  >
                    Login
                  </button>
                </div>

                <p className="text-center text-sm text-gray-500 mt-6 bg-gray-50 rounded-lg p-3">

                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

const API_BASE = "http://localhost:8000";

const TeacherDashboard = ({ email, onLogout }) => {
  const [assignments, setAssignments] = useState([]);
  const [teacherId, setTeacherId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [selectedAssignment, setSelectedAssignment] = useState(null);

  const [newAssignment, setNewAssignment] = useState({
    title: "",
    subject: "",
    description: "",
    startDate: "",
    deadline: "",
    latePenalty: 10,
    answerKeyFile: null
  });

  useEffect(() => {
    fetchAssignments();
  }, []);

  // ---------------- FETCH ----------------
  const fetchAssignments = async () => {
    console.log("🚀 fetchAssignments started for email:", email);
    try {
      const res = await fetch(`${API_BASE}/api/teachers/by-email/${email}`);
      console.log("💬 fetch response status:", res.status);
      if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`);

      const teacher = await res.json();
      console.log("📄 Teacher fetched:", teacher);

      setTeacherId(teacher._id);
      setAssignments(teacher.assignments || []);
    } catch (err) {
      console.error("❌ fetchAssignments error:", err);
      setError(err.message);
    } finally {
      console.log("✅ fetchAssignments finished");
      setLoading(false);
    }
  };

  const handleCreateAssignment = async () => {
    console.log("🚀 handleCreateAssignment started");

    if (!newAssignment.answerKeyFile) {
      console.warn("⚠️ No answer key file selected");
      setError("Answer key file is required!");
      return;
    }

    const formData = new FormData();
    formData.append("title", newAssignment.title);
    formData.append("subject", newAssignment.subject);
    formData.append("startDate", newAssignment.startDate + "T00:00:00Z");
    formData.append("deadline", newAssignment.deadline + "T23:59:59Z");
    formData.append("file", newAssignment.answerKeyFile);

    console.log("📦 FormData prepared:", Array.from(formData.entries()));

    try {
      const res = await fetch(`${API_BASE}/api/teachers/${teacherId}/assignments`, {
        method: "POST",
        body: formData,
      });

      console.log("💬 Response status:", res.status);

      if (!res.ok) {
        const errorText = await res.text();
        console.error("❌ Error response text:", errorText);
        throw new Error(`Create failed: ${res.status} ${errorText}`);
      }

      const created = await res.json();
      console.log("✅ Assignment created successfully:", created);

      setAssignments(prev => [created, ...prev]);
      setShowCreateForm(false);
      setNewAssignment({
        title: "",
        subject: "",
        description: "",
        startDate: "",
        deadline: "",
        latePenalty: 10,
        answerKeyFile: null
      });
    } catch (err) {
      console.error("❌ handleCreateAssignment error:", err);
      setError(err.message);
    }
  };

  // ---------------- PUBLISH ----------------
  const handlePublishResults = async (assignmentId) => {
    try {
      await fetch(
        `${API_BASE}/api/teachers/${teacherId}/assignments/${assignmentId}/publish`,
        { method: "PATCH" }
      );
      setAssignments(prev =>
        prev.map(a =>
          a.id === assignmentId ? { ...a, published: true } : a
        )
      );
    } catch {
      setError("Failed to publish results");
    }
  };

  if (loading) return <div className="p-6">Loading...</div>;
  if (error) return <div className="p-6 text-red-600">{error}</div>;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* HEADER */}

      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BookOpen className="w-8 h-8" />
            <h1 className="text-2xl font-bold">Teacher Dashboard</h1>
          </div>
          <button
            onClick={onLogout}
            className="flex items-center gap-2 bg-white/20 hover:bg-white/30 px-4 py-2 rounded-xl transition-all"
          >
            <LogOut className="w-4 h-4" />
            Logout
          </button>
        </div>
      </div>


      {/* BODY */}
      <div className="max-w-7xl mx-auto p-6">
        <button
          onClick={() => setShowCreateForm(true)}
          className="mb-6 bg-gradient-to-r from-green-500 to-green-600 text-white px-6 py-3 rounded-lg font-semibold flex items-center gap-2"
        >
          <Upload className="w-5 h-5" />
          Create Assignment
        </button>

        {/* CREATE MODAL */}
        {showCreateForm && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl w-full max-w-xl">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-5 rounded-t-xl">
                <h2 className="text-xl font-bold">Create Assignment</h2>
              </div>

              <div className="p-6 space-y-4">
                <input
                  placeholder="Title"
                  className="w-full border p-2 rounded"
                  value={newAssignment.title}
                  onChange={e =>
                    setNewAssignment({ ...newAssignment, title: e.target.value })
                  }
                />
                <input
                  placeholder="Subject"
                  className="w-full border p-2 rounded"
                  value={newAssignment.subject}
                  onChange={e =>
                    setNewAssignment({ ...newAssignment, subject: e.target.value })
                  }
                />
                <textarea
                  placeholder="Description"
                  className="w-full border p-2 rounded"
                  rows={3}
                  value={newAssignment.description}
                  onChange={e =>
                    setNewAssignment({ ...newAssignment, description: e.target.value })
                  }
                />
                Start Date
                <input
                  type="date"
                  className="w-full border p-2 rounded"
                  onChange={e =>
                    setNewAssignment({ ...newAssignment, startDate: e.target.value })
                  }
                />
                End Date
                <input
                  type="date"
                  className="w-full border p-2 rounded"
                  onChange={e =>
                    setNewAssignment({ ...newAssignment, deadline: e.target.value })
                  }
                />

                <input
                  type="file"
                  onChange={e =>
                    setNewAssignment({ ...newAssignment, answerKeyFile: e.target.files[0] })
                  }
                />
                <div className="flex gap-3">
                  <button
                    onClick={handleCreateAssignment}
                    className="flex-1 bg-blue-600 text-white py-2 rounded"
                  >
                    Create
                  </button>
                  <button
                    onClick={() => setShowCreateForm(false)}
                    className="flex-1 bg-gray-200 py-2 rounded"
                  >
                    Cancel
                  </button>
                </div>
              </div>

            </div>

          </div>
        )}

        {/* ASSIGNMENTS */}
        <div className="space-y-4">
          {assignments.map(a => (
            <div
              key={a.id}
              className="bg-white p-6 rounded-xl shadow cursor-pointer"
              onClick={() => setSelectedAssignment(a)}
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-xl font-bold">{a.title}</h3>
                  <p className="text-sm text-gray-500">{a.subject}</p>

                  <div className="grid grid-cols-2 gap-4 mt-4 text-sm text-gray-600">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4" />
                      {new Date(a.deadline).toLocaleString([], {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </div>
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4" />
                      Submissions: {a.submissions}
                    </div>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4" />
                      Evaluated: {a.evaluated}
                    </div>

                  </div>
                </div>

                {!a.published && a.evaluated === a.submissions && a.submissions > 0 && (
                  <button
                    onClick={e => {
                      e.stopPropagation();
                      handlePublishResults(a.id);
                    }}
                    className="bg-green-600 text-white px-4 py-2 rounded flex items-center gap-2"
                  >
                    <Send className="w-4 h-4" />
                    Publish
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* ANALYTICS MODAL */}
        {selectedAssignment && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl w-full max-w-2xl p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">{selectedAssignment.title} Analytics</h2>
                <button
                  onClick={() => setSelectedAssignment(null)}
                  className="text-red-600 font-bold"
                >
                  Close
                </button>
              </div>
              {/* Example Analytics */}
              <p>Submissions: {selectedAssignment.submissions}</p>
              <p>Evaluated: {selectedAssignment.evaluated}</p>
              <p>Status: {selectedAssignment.published ? "Published" : "Not Published"}</p>
              {/* You can add charts/graphs here later */}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
const StudentDashboard = ({ user, onLogout }) => {
  const [assignments, setAssignments] = useState([]);
  const [studentInfo, setStudentInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [studentSubmissions, setStudentSubmissions] = useState({});
  const [uploadFiles, setUploadFiles] = useState({}); // track file per assignment

  useEffect(() => {
    fetchDashboard();
  }, []);

  // ===============================
  // FETCH DASHBOARD
  // ===============================
  const fetchDashboard = async () => {
    try {
      console.log("🧠 user:", user);

      const res = await fetch(`${API_BASE}/api/students/dashboard`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ studentId: user.id }),
      });

      console.log("⬅️ Dashboard status:", res.status);

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      console.log("✅ Dashboard data:", data);

      setStudentInfo(data.student);
      setAssignments(data.assignments || []);

      const map = {};
      for (const a of data.assignments || []) {
        map[a.assignmentId] = {
          status: a.studentStatus,
          marks: a.marks,
          feedback: a.feedback,
        };
      }
      setStudentSubmissions(map);
    } catch (err) {
      console.error("❌ fetchDashboard error:", err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ===============================
  // SUBMIT ASSIGNMENT
  // ===============================
  const handleStudentSubmission = async (assignmentId) => {
    const file = uploadFiles[assignmentId];
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    console.log("🚀 Submitting", assignmentId, file);

    const formData = new FormData();
    formData.append("studentId", user.id);
    formData.append("assignmentId", assignmentId);
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/students/submit-assignment`, {
        method: "POST",
        body: formData,
      });

      console.log("⬅️ Submit status:", res.status);

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      console.log("✅ Submit success:", data);

      setStudentSubmissions((prev) => ({
        ...prev,
        [assignmentId]: {
          ...(prev[assignmentId] || {}),
          status: "submitted",
        },
      }));

      setUploadFiles((prev) => ({
        ...prev,
        [assignmentId]: null,
      }));
    } catch (err) {
      console.error("❌ submit error:", err);
      setError(err.message);
    }
  };

  if (loading) return <div className="p-6">Loading...</div>;
  if (error) return <div className="p-6 text-red-600">{error}</div>;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* NAVBAR */}
      <nav className="bg-indigo-600 text-white p-4 flex justify-between">
        <div>
          <h1 className="text-xl font-bold">Student Dashboard</h1>
          {studentInfo && <p className="text-sm">Welcome, {studentInfo.name}</p>}
        </div>
        <button onClick={onLogout}>Logout</button>
      </nav>

      {/* ASSIGNMENTS */}
      <div className="p-6 space-y-4">
        {assignments.length === 0 && (
          <p className="text-gray-600">No assignments assigned yet.</p>
        )}

        {assignments.map((a) => {
          const submission = studentSubmissions[a.assignmentId];
          return (
            <div key={a.assignmentId} className="bg-white p-4 rounded shadow space-y-2">
              <h2 className="font-bold text-lg">{a.title}</h2>
              <p className="text-sm text-gray-600">{a.subject}</p>

              {submission ? (
                <p className="text-green-600">Status: {submission.status}</p>
              ) : (
                <p className="text-red-600">Status: not submitted</p>
              )}

              <input
                type="file"
                onChange={(e) =>
                  setUploadFiles((prev) => ({
                    ...prev,
                    [a.assignmentId]: e.target.files[0],
                  }))
                }
                className="mt-2"
              />

              <button
                onClick={() => handleStudentSubmission(a.assignmentId)}
                className="mt-2 bg-indigo-600 text-white px-4 py-2 rounded"
              >
                Submit
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
};
const App = () => {
  const [currentUser, setCurrentUser] = useState(null);
  const [userType, setUserType] = useState(null);
  const handleLogin = (user, type) => {
    setCurrentUser(user);
    setUserType(type);
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setUserType(null);
  };

  if (!currentUser) {
    return <LoginPage onLogin={handleLogin} />;
  }

  if (userType === 'teacher') {
    return (
      <TeacherDashboard
        user={currentUser}
        email={currentUser.email}      // ← Send Gmail
        onLogout={handleLogout}
      />
    );
  }

  if (userType === 'student') {
    return (
      <StudentDashboard
        user={currentUser}
        email={currentUser.email}      // ← Send Gmail
        onLogout={handleLogout}
      />
    );
  }

  return null;
};

export default App;