from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":
    print("Hello LangChain!")

    linkedin_profile_url = linkedin_lookup_agent(name="Cedric Ancellin")
    linkedin_data = scrape_linkedin_profile(
        linkedin_profile_url=linkedin_profile_url)

    # you can add more tools here {linkedin_information} {twitter_information}

    summary_template = """
         given the Linkedin information {information} about a person from I want you to create:
         1. a short summary
         2. two interesting facts about them
     """

    # add more input varibles to the array if needed  input_variables=["linkedin_information", "linkedin_twitter"]
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # add more input variables here {"linkedin_information": linkedin_data, twitter_information: twitter_data}
    print(chain.invoke(input={"information": linkedin_data}))
