from tools.save_company import (
    create_company_record,
    save_company
)

record = create_company_record(
    company_name="Macsteel",
    website_url="https://www.macsteel.co.za",
    industry="Steel",
    research={
        "services": [
            "Steel Distribution",
            "Fabrication"
        ]
    },
    architecture={
        "sitemap": [
            "Home",
            "Services",
            "Projects",
            "Contact"
        ]
    },
    verification={
        "confirmed_facts": [
            "Company Name"
        ]
    }
)

saved = save_company(record)

print(saved)