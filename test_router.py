from llm_router import create_router

router = create_router()

print("Providers loaded:")
for provider in router.provider_status():
    if provider["configured"]:
        print(f"✅ {provider['name']}")

print("\nTesting router...\n")

response = router.call(
    "Reply with exactly: Router is operational"
)

print(response)