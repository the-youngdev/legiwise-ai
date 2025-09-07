import sys
import os

# --- PATH CORRECTION ---
# This code adds the main project directory (the parent of 'tests') to the Python path.
# This is a robust way to ensure that the 'from app.drafting import ...' line works correctly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# --- END PATH CORRECTION ---

# Now the import will work correctly
from app.drafting import generate_draft

def run_all_tests():
    """
    Runs tests for all available templates in the drafting module.
    """
    print("--- Starting Drafting Module Test ---")

    # --- Test Case 1: Simple Legal Notice ---
    print("\n[1] Generating draft for 'Simple Legal Notice'...")
    
    sample_notice_details = {
        "Recipient Name": "John Doe",
        "Recipient Address": "123 Fake Street, Anytown, AN 12345",
        "Date": "September 6, 2025",
        "Subject": "Unpaid Invoices",
        "Sender Name": "Jane Smith",
        "Sender Address": "456 Real Avenue, Otherville, OT 67890",
        "Statement of Facts": "Despite multiple reminders sent on August 1st and August 15th, 2025, invoices #101 and #102, totaling INR 50,000, remain unpaid.",
        "Action Required": "clear the outstanding payment in full",
        "Notice Period (e.g., 15 days)": "15 days"
    }
    
    generated_notice = generate_draft("Simple Legal Notice", sample_notice_details)
    
    print("\n--- Generated Draft ---")
    print(generated_notice)
    print("--- End of Draft ---\n")

    if "Error:" not in generated_notice:
        print("✅ Test Passed: 'Simple Legal Notice' generated successfully.")
    else:
        print("❌ Test Failed: An error occurred during 'Simple Legal Notice' generation.")

    # --- Test Case 2: Basic Rental Agreement ---
    print("\n[2] Generating draft for 'Basic Rental Agreement'...")

    sample_agreement_details = {
        "Landlord Name": "Peter Jones",
        "Landlord Address": "789 Property Lane, Landlord City, LC 11223",
        "Tenant Name": "Mary Williams",
        "Tenant Address": "101 Tenant Drive, Renter Town, RT 44556",
        "Property Address": "Apt 5B, 101 Tenant Drive, Renter Town, RT 44556",
        "Lease Start Date": "October 1, 2025",
        "Lease End Date": "September 30, 2026",
        "Monthly Rent (INR)": "25,000",
        "Security Deposit (INR)": "50,000"
    }

    generated_agreement = generate_draft("Basic Rental Agreement", sample_agreement_details)

    print("\n--- Generated Draft ---")
    print(generated_agreement)
    print("--- End of Draft ---\n")

    if "Error:" not in generated_agreement:
        print("✅ Test Passed: 'Basic Rental Agreement' generated successfully.")
    else:
        print("❌ Test Failed: An error occurred during 'Basic Rental Agreement' generation.")

    print("\n--- All Tests Complete ---")


if __name__ == "__main__":
    run_all_tests()

