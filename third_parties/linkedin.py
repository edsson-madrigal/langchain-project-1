import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from LinkedIn profiles,
    Manually scrape the information from the LinkedIn profile"""

    # api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    # will hard code the api_endpoint for now
    api_endpoint = "https://gist.githubusercontent.com/edsson-madrigal/ecbcbfed86f39e52c36151c9243ddc12/raw/d87d447d2dbfaaa5ef624af3c7adf26fc5e85c1b/gistfile1.json"
    header_dic = {
        "Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    # response = requests.get(api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic)
    response = requests.get(
        api_endpoint
    )
    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
