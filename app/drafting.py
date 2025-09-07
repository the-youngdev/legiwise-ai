# In app/drafting.py

# This file contains templates for generating legal documents.
# The 'fields' dictionary maps user-facing labels to simple, robust template keys.

DRAFTING_TEMPLATES = {
    "Simple Legal Notice": {
        "fields": {
            "Recipient Name": "recipient_name",
            "Recipient Address": "recipient_address",
            "Date": "date",
            "Subject": "subject",
            "Sender Name": "sender_name",
            "Sender Address": "sender_address",
            "Statement of Facts": "statement_of_facts",
            "Action Required": "action_required",
            "Notice Period (e.g., 15 days)": "notice_period"
        },
        "template": """
LEGAL NOTICE
--------------------

To,
{recipient_name}
{recipient_address}

Date: {date}

**Subject: Notice regarding {subject}**

Dear {recipient_name},

Under instructions from and on behalf of my client, {sender_name}, residing at {sender_address}, I do hereby serve you with the following legal notice:

1. That my client states the following facts: {statement_of_facts}.

2. That you are hereby required to comply with the following: {action_required}, within a period of {notice_period} from the receipt of this legal notice, failing which my client will be forced to initiate appropriate legal proceedings against you at your own risk and cost.

3. That a copy of this notice has been retained in my office for record and further necessary action.

Awaiting your compliance.

Yours faithfully,


(Advocate)
"""
    },
    
    "Basic Rental Agreement": {
        "fields": {
            "Landlord Name": "landlord_name",
            "Landlord Address": "landlord_address",
            "Tenant Name": "tenant_name",
            "Tenant Address": "tenant_address",
            "Property Address": "property_address",
            "Lease Start Date": "lease_start_date",
            "Lease End Date": "lease_end_date",
            "Monthly Rent (INR)": "monthly_rent_inr",
            "Security Deposit (INR)": "security_deposit_inr"
        },
        "template": """
RENTAL AGREEMENT
--------------------

This Rental Agreement is made and entered into on this day, {lease_start_date}, by and between:

**THE LANDLORD:**
Name: {landlord_name}
Address: {landlord_address}

AND

**THE TENANT:**
Name: {tenant_name}
Address: {tenant_address}

**1. PROPERTY:**
The Landlord agrees to lease to the Tenant the property located at: {property_address}.

**2. TERM:**
The term of this lease shall be from {lease_start_date} to {lease_end_date}.

**3. RENT:**
The Tenant shall pay the Landlord a monthly rent of INR {monthly_rent_inr}, payable on or before the 5th day of each month.

**4. SECURITY DEPOSIT:**
The Tenant has paid a security deposit of INR {security_deposit_inr} to the Landlord. This deposit will be refunded to the Tenant at the end of the lease term, subject to deductions for any damages to the property.

IN WITNESS WHEREOF, the parties have executed this agreement as of the date first above written.


_________________________
{landlord_name} (Landlord)


_________________________
{tenant_name} (Tenant)
"""
    }
}


def generate_draft(template_name: str, details: dict) -> str:
    """
    Populates a selected template with user-provided details by mapping UI labels
    to template keys.

    Args:
        template_name: The name of the template to use.
        details: A dictionary where keys are the user-facing field labels.

    Returns:
        The formatted legal draft as a string.
    """
    template_info = DRAFTING_TEMPLATES.get(template_name)
    if not template_info:
        return "Error: Selected template not found."
    
    try:
        # Create the dictionary for formatting by mapping UI labels to template keys
        format_dict = {}
        field_mapping = template_info["fields"]
        for ui_label, template_key in field_mapping.items():
            if ui_label in details:
                format_dict[template_key] = details[ui_label]
            else:
                raise KeyError(f"The required field '{ui_label}' was not provided.")

        return template_info["template"].format(**format_dict)
    except KeyError as e:
        return f"Error: A required field is missing from the input: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

