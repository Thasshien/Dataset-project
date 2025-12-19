import mongoose from "mongoose";

const AssignmentSchema = new mongoose.Schema({
  id: { type: String, required: true }, // unique assignment id
  title: { type: String, required: true },
  subject: { type: String, required: true },
  description: { type: String },
  startDate: { type: Date, required: true },
  deadline: { type: Date, required: true },
  status: { type: String, default: "active" },
  submissions: { type: Number, default: 0 },
  evaluated: { type: Number, default: 0 },
  published: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now },
  updatedAt: { type: Date },
  answerKey: {
    filename: String,
    uploadedAt: Date,
    url: String, // Cloudinary URL
  },
  rubric: { type: mongoose.Schema.Types.Mixed },
});

const TeacherSchema = new mongoose.Schema({
    _id: { type: String },    
  email: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  role: { type: String, default: "teacher" },
  createdAt: { type: Date, default: Date.now },
  lastLogin: { type: Date },
  assignments: [AssignmentSchema],
});

export const Teacher = mongoose.model("Teacher", TeacherSchema);

