#!/usr/bin/env python3
"""
Exam Answer Key PDF Generator
Generates a professional answer key with mixed DESCRIPTIVE and TECHNICAL questions
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
    PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import datetime


def create_answer_key_pdf(filename="exam_answer_key.pdf"):
    """Generate a professional exam answer key PDF"""

    # Create PDF document
    doc = SimpleDocTemplate(
        filename, 
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )

    # Container for PDF elements
    elements = []

    # Custom styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'QuestionHeading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#003366'),
        spaceAfter=4,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'NormalText',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=4
    )

    rubric_header = ParagraphStyle(
        'RubricHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#333333'),
        fontName='Helvetica-Bold',
        spaceAfter=2
    )

    # ===== HEADER =====
    elements.append(Paragraph("EXAM ANSWER KEY", title_style))
    elements.append(Paragraph("Computer Science - Advanced Topics", styles['Normal']))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    # ===== QUESTION 1: DESCRIPTIVE =====
    elements.append(Paragraph("Question 1 [10 marks]", heading_style))
    elements.append(Paragraph(
        "<b>Question:</b> Explain the concept of ACID properties in database transactions. "
        "Discuss how each property ensures data consistency and provide real-world scenarios where violation of these properties could lead to problems.",
        normal_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    # Rubric for Q1
    rubric_data_q1 = [
        ['Trait', 'Weight', 'Description'],
        ['Concept Coverage', '40%', 'Comprehensive explanation of all 4 ACID properties (Atomicity, Consistency, Isolation, Durability)'],
        ['Real-World Application', '30%', 'Clear examples of transactions and scenarios where violations cause problems'],
        ['Logical Flow', '20%', 'Well-organized answer with clear connections between properties'],
        ['Clarity & Language', '10%', 'Clear writing, appropriate terminology usage']
    ]

    rubric_table_q1 = Table(rubric_data_q1, colWidths=[1.8*inch, 0.8*inch, 2.7*inch])
    rubric_table_q1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))

    elements.append(Paragraph("<b>Evaluation Rubric:</b>", rubric_header))
    elements.append(rubric_table_q1)
    elements.append(Spacer(1, 0.15*inch))

    # ===== QUESTION 2: TECHNICAL =====
    elements.append(Paragraph("Question 2 [8 marks]", heading_style))
    elements.append(Paragraph(
        "<b>Question:</b> Write a SQL query to find the top 5 departments by average salary, "
        "excluding departments with fewer than 10 employees. Include department name and average salary in results.",
        normal_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("<b>Expected Answer:</b>", rubric_header))
    answer_q2 = """
<font face="Courier" size="8">
SELECT d.dept_name, AVG(e.salary) AS avg_salary<br/>
FROM departments d<br/>
INNER JOIN employees e ON d.dept_id = e.dept_id<br/>
GROUP BY d.dept_id, d.dept_name<br/>
HAVING COUNT(e.emp_id) >= 10<br/>
ORDER BY avg_salary DESC<br/>
LIMIT 5;
</font>
    """
    elements.append(Paragraph(answer_q2, normal_style))

    elements.append(Spacer(1, 0.08*inch))
    elements.append(Paragraph("<b>Key Points:</b>", rubric_header))
    keywords_q2 = [
        "Correct JOIN syntax connecting departments and employees tables",
        "GROUP BY clause with both dept_id and dept_name",
        "HAVING clause for filtering groups by employee count (COUNT >= 10)",
        "ORDER BY with DESC for descending average salary",
        "LIMIT 5 to restrict result set",
        "Correct aggregation function AVG()"
    ]
    for i, keyword in enumerate(keywords_q2, 1):
        elements.append(Paragraph(f"• {keyword}", normal_style))

    elements.append(Spacer(1, 0.15*inch))

    # ===== QUESTION 3: DESCRIPTIVE =====
    elements.append(Paragraph("Question 3 [12 marks]", heading_style))
    elements.append(Paragraph(
        "<b>Question:</b> Discuss the differences between supervised and unsupervised learning. "
        "Provide at least two examples for each category and explain why certain problem types are better suited to each approach.",
        normal_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    rubric_data_q3 = [
        ['Trait', 'Weight', 'Description'],
        ['Fundamental Differences', '35%', 'Clear explanation of labeled vs unlabeled data, training approach differences'],
        ['Examples Provided', '35%', 'Minimum 2 valid examples each (supervised: classification, regression; unsupervised: clustering, dimensionality reduction)'],
        ['Problem Suitability Analysis', '20%', 'Reasoning for why certain problems fit each approach'],
        ['Organization & Clarity', '10%', 'Well-structured response with clear delineation between sections']
    ]

    rubric_table_q3 = Table(rubric_data_q3, colWidths=[1.8*inch, 0.8*inch, 2.7*inch])
    rubric_table_q3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))

    elements.append(Paragraph("<b>Evaluation Rubric:</b>", rubric_header))
    elements.append(rubric_table_q3)
    elements.append(Spacer(1, 0.15*inch))

    # ===== QUESTION 4: TECHNICAL =====
    elements.append(Paragraph("Question 4 [7 marks]", heading_style))
    elements.append(Paragraph(
        "<b>Question:</b> Given an array of integers, implement an algorithm to find if there exists "
        "a subarray with sum equal to a target value. Time complexity should not exceed O(n).",
        normal_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("<b>Expected Answer Approach:</b>", rubric_header))
    answer_q4 = """
<font face="Courier" size="8">
def find_subarray_sum(arr, target):<br/>
&nbsp;&nbsp;&nbsp;&nbsp;seen_sums = {0}  # Set to track cumulative sums<br/>
&nbsp;&nbsp;&nbsp;&nbsp;current_sum = 0<br/>
&nbsp;&nbsp;&nbsp;&nbsp;<br/>
&nbsp;&nbsp;&nbsp;&nbsp;for num in arr:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;current_sum += num<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if (current_sum - target) in seen_sums:<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return True<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;seen_sums.add(current_sum)<br/>
&nbsp;&nbsp;&nbsp;&nbsp;return False
</font>
    """
    elements.append(Paragraph(answer_q4, normal_style))

    elements.append(Spacer(1, 0.08*inch))
    elements.append(Paragraph("<b>Key Points:</b>", rubric_header))
    keywords_q4 = [
        "Use of hash set/dictionary for O(1) lookup of cumulative sums",
        "Cumulative sum approach to track running total",
        "Logic: if (current_sum - target) exists, then subarray found",
        "Single pass through array = O(n) time complexity",
        "O(n) space complexity for set storage",
        "Handles negative numbers and edge cases",
        "Alternative: Sliding window for positive integers only"
    ]
    for keyword in keywords_q4:
        elements.append(Paragraph(f"• {keyword}", normal_style))

    elements.append(Spacer(1, 0.15*inch))

    # ===== QUESTION 5: DESCRIPTIVE =====
    elements.append(Paragraph("Question 5 [9 marks]", heading_style))
    elements.append(Paragraph(
        "<b>Question:</b> Analyze the impact of network latency and bandwidth constraints on distributed system design. "
        "How would you architect a system to handle high-latency and low-bandwidth environments?",
        normal_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    rubric_data_q5 = [
        ['Trait', 'Weight', 'Description'],
        ['Impact Analysis', '30%', 'Explanation of latency/bandwidth effects on system performance and reliability'],
        ['Design Strategies', '40%', 'Multiple strategies (caching, compression, async communication, local processing, batching)'],
        ['Trade-offs Discussion', '20%', 'Understanding of consistency vs performance trade-offs'],
        ['Coherence & Examples', '10%', 'Well-articulated with practical examples']
    ]

    rubric_table_q5 = Table(rubric_data_q5, colWidths=[1.8*inch, 0.8*inch, 2.7*inch])
    rubric_table_q5.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))

    elements.append(Paragraph("<b>Evaluation Rubric:</b>", rubric_header))
    elements.append(rubric_table_q5)
    elements.append(Spacer(1, 0.15*inch))

    # ===== QUESTION 6: TECHNICAL =====
    elements.append(Paragraph("Question 6 [6 marks]", heading_style))
    elements.append(Paragraph(
        "<b>Question:</b> Write a MongoDB aggregation pipeline to calculate the monthly revenue "
        "for each product category, sorting by revenue in descending order.",
        normal_style
    ))
    elements.append(Spacer(1, 0.1*inch))

    elements.append(Paragraph("<b>Expected Answer:</b>", rubric_header))
    answer_q6 = """
<font face="Courier" size="8">
db.orders.aggregate([<br/>
&nbsp;&nbsp;&nbsp;&nbsp;{<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$group: {<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_id: {<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;category: "$product.category",<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;month: { $dateToString: { format: "%Y-%m", date: "$order_date" } }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;monthly_revenue: { $sum: "$total_amount" }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}<br/>
&nbsp;&nbsp;&nbsp;&nbsp;},<br/>
&nbsp;&nbsp;&nbsp;&nbsp;{<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$sort: { monthly_revenue: -1 }<br/>
&nbsp;&nbsp;&nbsp;&nbsp;}<br/>
])
</font>
    """
    elements.append(Paragraph(answer_q6, normal_style))

    elements.append(Spacer(1, 0.08*inch))
    elements.append(Paragraph("<b>Key Points:</b>", rubric_header))
    keywords_q6 = [
        "Correct $group stage with nested _id for category and month",
        "$dateToString for formatting date to YYYY-MM format",
        "$sum aggregation to calculate total revenue",
        "$sort stage with -1 for descending order",
        "Proper MongoDB syntax and nested field access ($product.category)",
        "Handles multiple documents per category-month combination",
        "Alternative: Using $month and $year functions instead of $dateToString"
    ]
    for keyword in keywords_q6:
        elements.append(Paragraph(f"• {keyword}", normal_style))

    elements.append(Spacer(1, 0.2*inch))

    # ===== FOOTER =====
    elements.append(PageBreak())
    elements.append(Paragraph("GRADING GUIDELINES", heading_style))
    elements.append(Spacer(1, 0.1*inch))

    footer_data = [
        ['Question', 'Type', 'Max Marks', 'Grading Method'],
        ['1', 'DESCRIPTIVE', '10', 'Rubric-based with weighted traits'],
        ['2', 'TECHNICAL', '8', 'Correctness + Code quality + Keywords'],
        ['3', 'DESCRIPTIVE', '12', 'Rubric-based with weighted traits'],
        ['4', 'TECHNICAL', '7', 'Algorithm correctness + Complexity analysis'],
        ['5', 'DESCRIPTIVE', '9', 'Rubric-based with weighted traits'],
        ['6', 'TECHNICAL', '6', 'Query correctness + Syntax + Stage usage'],
        ['', '', '<b>Total: 52</b>', '']
    ]

    footer_table = Table(footer_data, colWidths=[1*inch, 1.8*inch, 1.2*inch, 2.2*inch])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))

    elements.append(footer_table)
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(
        "<b>Note:</b> This answer key is for instructor reference only. "
        "Students should demonstrate understanding of concepts, proper methodology, and clear communication. "
        "Partial credit may be awarded for partially correct answers.",
        ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, textColor=colors.grey, italic=True)
    ))

    # Build PDF
    doc.build(elements)
    print(f"✅ PDF generated successfully: {filename}")
    return filename


if __name__ == "__main__":
    create_answer_key_pdf("exam_answer_key.pdf")
