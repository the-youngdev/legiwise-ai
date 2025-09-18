# In app/drafting.py

DRAFTING_TEMPLATES = {
    "Simple Legal Notice": {
        "fields": {
            "Recipient Name": "John Doe",
            "Recipient Address": "123 Fake Street, Anytown, AN 12345",
            "Date": "September 19, 2025",
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
    },
    "General Power of Attorney": {
        "fields": {
            "Principal Name": "Anjali Sharma",
            "Principal Address": "Flat 1A, Green Apartments, Kolkata",
            "Agent Name": "Rohan Sharma",
            "Agent Address": "Flat 1A, Green Apartments, Kolkata",
            "Date of Execution": "September 19, 2025",
            "Place of Execution": "Kolkata",
            "Powers Granted": "To manage my bank accounts, sign legal documents, and represent me in all property-related matters."
        },
        "template": """
GENERAL POWER OF ATTORNEY
---------------------------
KNOW ALL MEN BY THESE PRESENTS, that I, {Principal Name}, residing at {Principal Address}, do hereby appoint {Agent Name}, residing at {Agent Address}, as my true and lawful attorney-in-fact (Agent).

I hereby grant my Agent the full power and authority to do and perform all and every act and thing whatsoever requisite and necessary to be done in and about the premises, as fully to all intents and purposes as I might or could do if personally present, including but not limited to:

{Powers Granted}

This Power of Attorney shall be effective from the date of execution.

IN WITNESS WHEREOF, I have hereunto set my hand on this {Date of Execution} at {Place of Execution}.

_________________________
({Principal Name}) - Principal
"""
    },
    "Cease and Desist Letter (Defamation)": {
        "fields": {
            "Recipient Name": "Online News Portal Inc.",
            "Recipient Address": "Digital Media Center, Mumbai",
            "Date": "September 19, 2025",
            "Sender Name": "Vikram Singh",
            "Infringing Content Description": "the article titled 'Local Businessman in Scandal' published on your website on September 15, 2025",
            "Specific False Statements": "'Mr. Singh was involved in illegal activities'",
            "Action Required": "immediately remove the aforementioned article and publish a formal retraction and apology",
            "Deadline": "7 days"
        },
        "template": """
CEASE AND DESIST
--------------------
To,
{Recipient Name}
{Recipient Address}

Date: {Date}

**Subject: Demand to Cease and Desist from Defamatory Statements**

Dear {Recipient Name},

This letter is a formal demand that you immediately cease and desist from making and publishing false and defamatory statements against me, {Sender Name}.

It has come to my attention that you have published {Infringing Content Description}, which contains verifiably false statements, including but not to: {Specific False Statements}.

These statements are untrue, harmful to my reputation, and constitute defamation under Indian law.

I demand that you {Action Required} within {Deadline} of the receipt of this letter.

Failure to comply will result in me pursuing all available legal remedies against you.

Sincerely,
{Sender Name}
"""
    },
    "Simple Non-Disclosure Agreement (NDA)": {
        "fields": {
            "Disclosing Party Name": "Innovate Solutions Pvt. Ltd.",
            "Receiving Party Name": "Creative Designs Co.",
            "Effective Date": "September 19, 2025",
            "Confidential Information Description": "business plans, financial data, customer lists, and proprietary software related to 'Project Phoenix'"
        },
        "template": """
NON-DISCLOSURE AGREEMENT (NDA)
----------------------------------
This Agreement is made on {Effective Date} between {Disclosing Party Name} ("Disclosing Party") and {Receiving Party Name} ("Receiving Party").

1.  **Confidential Information:** The Disclosing Party may disclose to the Receiving Party certain confidential and proprietary information, specifically: {Confidential Information Description}.

2.  **Obligations:** The Receiving Party agrees not to disclose the Confidential Information to any third party and to use it solely for the purpose of evaluating a potential business relationship with the Disclosing Party.

3.  **Term:** This Agreement shall remain in effect until the Confidential Information is no longer a trade secret or until the Disclosing Party sends the Receiving Party written notice releasing them from this Agreement.

Signed,

_________________________
{Disclosing Party Name}

_________________________
{Receiving Party Name}
"""
    },
    "Affidavit for Change of Name": {
        "fields": {
            "Deponent Name (Old Name)": "Priya Kumari",
            "Deponent Name (New Name)": "Priya Sharma",
            "Father's/Husband's Name": "Ravi Sharma",
            "Deponent Address": "7B, Park Street, Kolkata",
            "Date": "September 19, 2025",
            "Place": "Kolkata"
        },
        "template": """
AFFIDAVIT
-----------
I, {Deponent Name (New Name)}, formerly {Deponent Name (Old Name)}, daughter/wife of {Father's/Husband's Name}, residing at {Deponent Address}, do hereby solemnly affirm and declare as under:

1. That my old name is {Deponent Name (Old Name)}.
2. That my new name is {Deponent Name (New Name)}.
3. That I shall henceforth be known as {Deponent Name (New Name)} for all purposes.

I declare that the statements made above are true to the best of my knowledge and belief.

_________________________
(Deponent)

Verification:
Verified at {Place} on this {Date} that the contents of this affidavit are true and correct.

_________________________
(Deponent)
"""
    },
    "Last Will and Testament": {
        "fields": {
            "Testator Name": "Rajesh Gupta",
            "Testator Address": "15, Ballygunge Circular Road, Kolkata",
            "Date": "September 19, 2025",
            "Executor Name": "Sunita Gupta",
            "Executor Address": "15, Ballygunge Circular Road, Kolkata",
            "Beneficiary 1 Name": "Amit Gupta (Son)",
            "Bequest to Beneficiary 1": "my residential property located at 15, Ballygunge Circular Road, Kolkata",
            "Beneficiary 2 Name": "Anita Roy (Daughter)",
            "Bequest to Beneficiary 2": "all my shares in ABC Corp. and my entire bank balance in XYZ Bank"
        },
        "template": """
LAST WILL AND TESTAMENT
---------------------------
I, {Testator Name}, residing at {Testator Address}, being of sound mind and memory, do hereby make, publish and declare this to be my last Will and Testament.

1.  I hereby revoke all former wills and codicils made by me.

2.  I appoint {Executor Name} of {Executor Address} to be the Executor of this Will.

3.  I give, devise, and bequeath my property as follows:
    a) To {Beneficiary 1 Name}, I bequeath {Bequest to Beneficiary 1}.
    b) To {Beneficiary 2 Name}, I bequeath {Bequest to Beneficiary 2}.

IN WITNESS WHEREOF, I have signed this Will on {Date}.

_________________________
({Testator Name})

WITNESSES:
1. _________________________
2. _________________________
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