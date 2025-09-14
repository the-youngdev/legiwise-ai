# In app/drafting.py

DRAFTING_TEMPLATES = {
    "Simple Legal Notice": {
        "fields": {
            "Recipient Name": "John Doe",
            "Recipient Address": "123 Fake Street, Anytown, AN 12345",
            "Date": "September 7, 2025",
            "Subject": "Unpaid Invoices",
            "Sender Name": "Jane Smith",
            "Sender Address": "456 Real Avenue, Otherville, OT 67890",
            "Statement of Facts": "Despite multiple reminders...",
            "Action Required": "clear the outstanding payment in full",
            "Notice Period (e.g., 15 days)": "15 days"
        },
        "template": """
LEGAL NOTICE
--------------------
To,
{Recipient Name}
{Recipient Address}

Date: {Date}

**Subject: Notice regarding {Subject}**

Dear {Recipient Name},

Under instructions from and on behalf of my client, {Sender Name}, residing at {Sender Address}, I hereby serve you with the following legal notice:

1. That my client states the following facts: {Statement of Facts}.

2. That you are hereby required to comply with the following: {Action Required}, within a period of {Notice Period (e.g., 15 days)} from the receipt of this notice.

Awaiting your compliance.

Yours faithfully,
(Advocate)
"""
    },
    "Basic Rental Agreement": {
        "fields": {
            "Landlord Name": "Peter Jones",
            "Landlord Address": "789 Property Lane, LC 11223",
            "Tenant Name": "Mary Williams",
            "Tenant Address": "101 Tenant Drive, RT 44556",
            "Property Address": "Apt 5B, 101 Tenant Drive, RT 44556",
            "Lease Start Date": "October 1, 2025",
            "Lease End Date": "September 30, 2026",
            "Monthly Rent (INR)": "25,000",
            "Security Deposit (INR)": "50,000"
        },
        "template": """
RENTAL AGREEMENT
--------------------
This agreement is made on {Lease Start Date} between {Landlord Name} (Landlord) and {Tenant Name} (Tenant).

The Landlord agrees to lease to the Tenant the property at: {Property Address}.

The term of this lease is from {Lease Start Date} to {Lease End Date}.

The Tenant shall pay a monthly rent of INR {Monthly Rent (INR)}.

A security deposit of INR {Security Deposit (INR)} has been paid.

IN WITNESS WHEREOF, the parties have executed this agreement.

_________________________
{Landlord Name} (Landlord)

_________________________
{Tenant Name} (Tenant)
"""
    }
}

def generate_draft(template_name: str, details: dict) -> str:
    """Populates a template with user-provided details."""
    template_info = DRAFTING_TEMPLATES.get(template_name)
    if not template_info:
        return "Error: Template not found."
    try:
        return template_info["template"].format(**details)
    except KeyError as e:
        return f"Error: Missing field in details: {e}"