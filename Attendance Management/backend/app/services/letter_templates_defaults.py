"""Default HR letter templates. Ensured on first list so Letter templates are never empty."""
from sqlalchemy.orm import Session
from app.models import LetterTemplate

# Placeholders: employee_name, employee_code, date_of_joining, designation, department, date, basic, hra, allowances, gross
DEFAULT_TEMPLATES = [
    {
        "code": "CONFIRM",
        "name": "Confirmation Letter",
        "subject_template": "Confirmation of Employment - {{ employee_name }}",
        "body_template": """<p>Date: {{ date.strftime('%d-%m-%Y') }}</p>
<p>To,<br/>{{ employee_name }}<br/>Employee Code: {{ employee_code }}</p>
<p>Subject: <strong>Confirmation of Employment</strong></p>
<p>Dear {{ employee_name }},</p>
<p>We are pleased to confirm that your services have been found satisfactory during the probation period. You are hereby confirmed in the position of <strong>{{ designation }}</strong> in the <strong>{{ department }}</strong> department with effect from {{ date_of_joining.strftime('%d-%m-%Y') }}.</p>
<p>Your confirmation is subject to the company's policies and terms of employment. We look forward to your continued contribution.</p>
<p>Yours sincerely,<br/>Human Resources</p>""",
    },
    {
        "code": "OFFER",
        "name": "Offer Letter",
        "subject_template": "Offer of Employment - {{ employee_name }}",
        "body_template": """<p>Date: {{ date.strftime('%d-%m-%Y') }}</p>
<p>To,<br/>{{ employee_name }}</p>
<p>Subject: <strong>Offer of Employment</strong></p>
<p>Dear {{ employee_name }},</p>
<p>We are pleased to offer you the position of <strong>{{ designation }}</strong> in the <strong>{{ department }}</strong> department. You are required to join on <strong>{{ date_of_joining.strftime('%d-%m-%Y') }}</strong>.</p>
<p>Your Employee Code will be: <strong>{{ employee_code }}</strong>. Please refer to the attached terms and conditions. Please sign and return a copy of this letter to confirm your acceptance.</p>
<p>We look forward to welcoming you.</p>
<p>Yours sincerely,<br/>Human Resources</p>""",
    },
    {
        "code": "RELIEVING",
        "name": "Relieving Letter",
        "subject_template": "Relieving Letter - {{ employee_name }}",
        "body_template": """<p>Date: {{ date.strftime('%d-%m-%Y') }}</p>
<p>To whomsoever it may concern,</p>
<p>This is to certify that <strong>{{ employee_name }}</strong> (Employee Code: {{ employee_code }}) was employed with our organisation as <strong>{{ designation }}</strong> in the <strong>{{ department }}</strong> department from <strong>{{ date_of_joining.strftime('%d-%m-%Y') }}</strong> till the date of relieving.</p>
<p>He/She has resigned from the company and has been relieved from all duties. We wish him/her success in future endeavours.</p>
<p>Yours sincerely,<br/>Human Resources</p>""",
    },
    {
        "code": "TERMINATION",
        "name": "Termination Letter",
        "subject_template": "Termination of Employment - {{ employee_name }}",
        "body_template": """<p>Date: {{ date.strftime('%d-%m-%Y') }}</p>
<p>To,<br/>{{ employee_name }}<br/>Employee Code: {{ employee_code }}</p>
<p>Subject: <strong>Termination of Employment</strong></p>
<p>Dear {{ employee_name }},</p>
<p>This letter is to inform you that your employment with the company in the capacity of <strong>{{ designation }}</strong>, <strong>{{ department }}</strong>, is being terminated with effect from the date of this letter. You are required to complete the handover and exit formalities as communicated separately.</p>
<p>Please contact HR for any clarifications regarding settlement and exit process.</p>
<p>Yours sincerely,<br/>Human Resources</p>""",
    },
    {
        "code": "INCREMENT",
        "name": "Increment Letter",
        "subject_template": "Salary Increment - {{ employee_name }}",
        "body_template": """<p>Date: {{ date.strftime('%d-%m-%Y') }}</p>
<p>To,<br/>{{ employee_name }}<br/>Employee Code: {{ employee_code }}</p>
<p>Subject: <strong>Revision of Salary (Increment)</strong></p>
<p>Dear {{ employee_name }},</p>
<p>We are pleased to inform you that based on your performance and contribution as <strong>{{ designation }}</strong> in <strong>{{ department }}</strong>, your salary has been revised. The revised structure is as under (effective as per company policy):</p>
<p>Basic: Rs. {{ basic }} | HRA: Rs. {{ hra }} | Allowances: Rs. {{ allowances }} | <strong>Gross: Rs. {{ gross }}</strong></p>
<p>We appreciate your dedication and look forward to your continued commitment.</p>
<p>Yours sincerely,<br/>Human Resources</p>""",
    },
    {
        "code": "SALARY_SLIP",
        "name": "Salary Slip",
        "subject_template": "Salary Slip - {{ employee_name }} - {{ date.strftime('%B %Y') }}",
        "body_template": """<p style="text-align:center;"><strong>SALARY SLIP</strong><br/>{{ date.strftime('%B %Y') }}</p>
<p><strong>Employee Name:</strong> {{ employee_name }}<br/><strong>Employee Code:</strong> {{ employee_code }}<br/><strong>Designation:</strong> {{ designation }}<br/><strong>Department:</strong> {{ department }}<br/><strong>Date of Joining:</strong> {{ date_of_joining.strftime('%d-%m-%Y') }}</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
<tr><th>Earnings</th><th>Amount (Rs.)</th></tr>
<tr><td>Basic</td><td>{{ basic }}</td></tr>
<tr><td>HRA</td><td>{{ hra }}</td></tr>
<tr><td>Allowances</td><td>{{ allowances }}</td></tr>
<tr><th>Gross Salary</th><th>{{ gross }}</th></tr>
</table>
<p><em>This is a system-generated slip. For deductions and net pay, refer to the official payslip from Payroll.</em></p>""",
    },
    {
        "code": "INTERNSHIP_6M",
        "name": "6 Months Internship Completion Certificate",
        "subject_template": "Internship Completion Certificate - {{ employee_name }}",
        "body_template": """<p style="text-align:center;"><strong>INTERNSHIP COMPLETION CERTIFICATE</strong></p>
<p>Date: {{ date.strftime('%d-%m-%Y') }}</p>
<p>This is to certify that <strong>{{ employee_name }}</strong> (Employee Code: {{ employee_code }}) has successfully completed the <strong>Six (6) Months Internship</strong> programme with our organisation in the <strong>{{ department }}</strong> department as <strong>{{ designation }}</strong>.</p>
<p>Internship period: From {{ date_of_joining.strftime('%d-%m-%Y') }} to the date of completion. During this period, he/she has been sincere, dedicated and has completed the assigned tasks to our satisfaction.</p>
<p>We wish him/her success in his/her career.</p>
<p>Yours sincerely,<br/>Human Resources</p>""",
    },
    {
        "code": "EXPERIENCE",
        "name": "Experience Letter",
        "subject_template": "Experience Certificate - {{ employee_name }}",
        "body_template": """<p>Date: {{ date.strftime('%d-%m-%Y') }}</p>
<p>To whomsoever it may concern,</p>
<p>This is to certify that <strong>{{ employee_name }}</strong> (Employee Code: {{ employee_code }}) was employed with our organisation as <strong>{{ designation }}</strong> in the <strong>{{ department }}</strong> department from <strong>{{ date_of_joining.strftime('%d-%m-%Y') }}</strong>.</p>
<p>During the tenure, he/she has been responsible, diligent and has performed his/her duties to our satisfaction. We wish him/her all the best for future endeavours.</p>
<p>Yours sincerely,<br/>Human Resources</p>""",
    },
]


def ensure_default_letter_templates(db: Session) -> None:
    """Create default letter templates if they do not exist (by code)."""
    existing_codes = {t.code for t in db.query(LetterTemplate).all()}
    for defn in DEFAULT_TEMPLATES:
        if defn["code"] in existing_codes:
            continue
        t = LetterTemplate(
            code=defn["code"],
            name=defn["name"],
            subject_template=defn["subject_template"],
            body_template=defn["body_template"],
            is_editable=True,
        )
        db.add(t)
        existing_codes.add(defn["code"])
    db.commit()
