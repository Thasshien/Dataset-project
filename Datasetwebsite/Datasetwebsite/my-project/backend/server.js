import express from "express";
import mongoose from "mongoose";
import cors from "cors";
import multer from "multer";
import { v4 as uuidv4 } from "uuid";
import fs from "fs";
import cloudinary from "cloudinary";
import { Teacher } from "./models/Teacher.js"; // use .js
import { Student } from "./models/student.js"; // use .js

const app = express();
app.use(cors());
app.use(express.json());

// ---------------- MONGODB ----------------

const MONGO_URI = "mongodb+srv://daktrboys05_db_user:gdgclubproject@to-do-list.qmqixqe.mongodb.net/tries_db";

mongoose.connect(MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.error("MongoDB connection error:", err));


// ---------------- CLOUDINARY ----------------
cloudinary.v2.config({
  cloud_name: "deybojppm",
  api_key: "918833843129511",
  api_secret: "zSa0IqGtZzwIOyqJyAn6Ng-Nrss",
});

// ---------------- MULTER ----------------
const upload = multer({ dest: "uploads/" });

// ---------------- ROUTES ----------------

// GET teacher by email
app.get("/api/teachers/by-email/:email", async (req, res) => {
  const { email } = req.params;
  console.log("🔍 GET /teachers/by-email called with:", email);
  try {
    const teacher = await Teacher.findOne({ email });
    console.log("📄 Teacher found:", teacher);
    if (!teacher) return res.status(404).json({ detail: "Teacher not found" });
    res.json(teacher);
  } catch (err) {
    console.error("❌ GET /teachers/by-email error:", err);
    res.status(500).json({ detail: "Server error" });
  }
});

app.post(
  "/api/teachers/:teacherId/assignments",
  upload.single("file"),
  async (req, res) => {
    const { teacherId } = req.params;

    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    console.log("🚀 POST /teachers/:teacherId/assignments called");
    console.log("➡️ teacherId:", teacherId);
    console.log("📦 req.body:", req.body);
    console.log("📄 req.file:", req.file);
    console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    try {
      if (!teacherId) {
        return res.status(400).json({ detail: "teacherId is required" });
      }

      // Fetch teacher from DB
      const teacher = await Teacher.findById(teacherId);
      if (!teacher) {
        console.warn("⚠️ Teacher not found for id:", teacherId);
        return res.status(404).json({ detail: "Teacher not found" });
      }

      const { title, subject, startDate, deadline, description } = req.body;

      if (!title || !subject || !startDate || !deadline) {
        return res.status(400).json({ detail: "Missing required fields" });
      }

      if (!req.file) {
        console.warn("⚠️ No file uploaded");
        return res.status(400).json({ detail: "Answer key file is required" });
      }

      // Upload file to Cloudinary
      console.log("⬆️ Uploading file to Cloudinary:", req.file.path);
      const result = await cloudinary.v2.uploader.upload(req.file.path, {
        folder: "teacher_assignments",
      });
      fs.unlinkSync(req.file.path); // remove local file
      console.log("✅ File uploaded:", result.secure_url);

      // Construct assignment object
      const assignment = {
        _id: new mongoose.Types.ObjectId(), // MongoDB ObjectId
        id: uuidv4(), // custom UUID for frontend reference
        title,
        subject,
        description: description || "",
        startDate: new Date(startDate),
        deadline: new Date(deadline),
        status: "active",
        submissions: 0,
        evaluated: 0,
        published: false,
        createdAt: new Date(),
        answerKey: {
          filename: req.file.originalname,
          url: result.secure_url,
          uploadedAt: new Date(),
        },
      };

      // Add to teacher assignments
      teacher.assignments.unshift(assignment);
      await teacher.save();
      console.log("✅ Assignment saved in teacher document:", assignment.id);

      res.json({ message: "Assignment created successfully", assignment });
    } catch (err) {
      console.error("❌ POST /assignments error:", err);
      res.status(500).json({ detail: "Server error" });
    }
  }
);
// Publish assignment
app.patch("/api/teachers/:teacherId/assignments/:assignmentId/publish", async (req, res) => {
  try {
    const { teacherId, assignmentId } = req.params;
    const teacher = await Teacher.findById(teacherId);
    if (!teacher) return res.status(404).json({ detail: "Teacher not found" });

    const assignment = teacher.assignments.find(a => a._id === assignmentId);
    if (!assignment) return res.status(404).json({ detail: "Assignment not found" });

    assignment.published = true;
    assignment.updatedAt = new Date();

    await teacher.save();
    res.json({ message: "Published" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ detail: "Server error" });
  }
});
// ---------------- STUDENT ROUTES ----------------

app.post("/api/students/dashboard", async (req, res) => {
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("📘 POST /api/students/dashboard");
  console.log("➡️ body:", req.body);
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

  try {
    const { studentId } = req.body;

    if (!studentId) {
      return res.status(400).json({ detail: "studentId is required" });
    }

    console.log("🔍 Finding student by _id:", studentId);

    const student = await Student.findById(studentId);

    if (!student) {
      console.warn("⚠️ Student not found for id:", studentId);
      return res.status(404).json({ detail: "Student not found" });
    }

    console.log("✅ Student found:", student.email);

    const resolvedAssignments = [];

    for (const sa of student.assignments) {
      const teacher = await Teacher.findById(sa.teacherId);
      if (!teacher) continue;

      const assignment = teacher.assignments.id(sa.assignmentId);
      if (!assignment) continue;

      resolvedAssignments.push({
        ...assignment.toObject(),
        assignmentId: assignment._id,
        teacherId: sa.teacherId,
        studentStatus: sa.status,
        submittedAt: sa.submittedAt,
        marks: sa.marks,
        feedback: sa.feedback || "",
      });
    }

    res.json({
      student: {
        id: student._id,
        name: student.name,
        email: student.email,
      },
      assignments: resolvedAssignments,
    });
  } catch (err) {
    console.error("❌ Dashboard error:", err);
    res.status(500).json({ detail: "Server error" });
  }
});

app.post("/api/students/submit-assignment", upload.single("file"), async (req, res) => {
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("📤 POST /submit-assignment (Cloudinary)");
  console.log("➡️ body:", req.body);
  console.log("➡️ file:", req.file);
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

  try {
    const { studentId, assignmentId } = req.body;

    if (!studentId || !assignmentId) {
      return res.status(400).json({ detail: "studentId and assignmentId are required" });
    }

    if (!req.file) {
      return res.status(400).json({ detail: "File is required" });
    }

    console.log("🔍 Fetching student by _id:", studentId);
    const student = await Student.findById(studentId);
    if (!student) {
      console.warn("⚠️ Student not found");
      return res.status(404).json({ detail: "Student not found" });
    }

    console.log("🔍 Finding assignment in student record");
    const sa = student.assignments.find(
      a => a.assignmentId.toString() === assignmentId
    );
    if (!sa) {
      console.warn("⚠️ Assignment not assigned to student");
      return res.status(404).json({ detail: "Assignment not assigned" });
    }

    // Upload file to Cloudinary
    console.log("⬆️ Uploading student submission to Cloudinary:", req.file.path);
    const result = await cloudinary.v2.uploader.upload(req.file.path, {
      folder: "student_submissions",
    });
    fs.unlinkSync(req.file.path); // remove local file

    // Save submission info
    // Save submission info
    sa.status = "submitted";
    sa.submittedAt = new Date();
    sa.answerFile = {
      filename: req.file.originalname,
      url: result.secure_url,
      uploadedAt: new Date(),
    };

    await student.save();
    console.log("✅ Submission saved in MongoDB:", sa.answerFile);


    console.log("✅ Submission saved:", {
      studentId,
      assignmentId,
      fileUrl: result.secure_url,
    });

    res.json({
      message: "Assignment submitted successfully",
      fileUrl: result.secure_url,
    });
  } catch (err) {
    console.error("❌ Submission error:", err);
    res.status(500).json({ detail: "Submission failed" });
  }
});


// ---------------- START SERVER ----------------
const PORT = 8000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
