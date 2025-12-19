import mongoose from "mongoose";

const StudentSchema = new mongoose.Schema({
  _id: { type: String }, // roll no / registration id
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },

  assignments: [
    {
      assignmentId: {
        type: mongoose.Schema.Types.ObjectId, // Teacher.assignments._id
        required: true,
      },
      teacherId: {
        type: String, // Teacher._id
        required: true,
      },

      status: {
        type: String,
        enum: ["assigned", "submitted", "evaluated"],
        default: "assigned",
      },

      assignedAt: {
        type: Date,
        default: Date.now,
      },

      submittedAt: {
        type: Date,
      },

      // 🔹 NEW: student answer upload info
      answerFile: {
        url: { type: String },        // Cloudinary secure_url
        filename: { type: String },   // original filename
        uploadedAt: { type: Date },
      },

      marks: Number,
    }
  ],

  createdAt: { type: Date, default: Date.now }
});

export const Student = mongoose.model("Student", StudentSchema);
