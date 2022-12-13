#!/usr/bin/env python
# coding: utf-8

# In[4]:


from bs4 import BeautifulSoup
import requests # For requesting info from a specific website
import time
import pandas as pd


# In[7]:


def find_jobs():
    unfamiliar_skills = input('Type skills that you are not familiar with ')
    print(f'Filtering out jobs requiring {unfamiliar_skills}')
    print()  
    #Input URL for page which came on searching for python jobs
    html_text = requests.get('https://www.timesjobs.com/candidate/job-search.html?searchType=personalizedSearch&from=submit&txtKeywords=Python&txtLocation=').text
    #Creating a Soup Instance with parser = lxml
    soup = BeautifulSoup(html_text,'lxml')
    jobs = soup.find_all('li',class_ = 'clearfix job-bx wht-shd-bx') #Searching with class_ keywork as Class is already a reserved keyword

    company_name_list = []
    skills_list = []
    more_info_list = []
    for index, job in enumerate(jobs):    
        published_date = job.find('span',class_ = 'sim-posted').span.text.strip()
        if 'few' not in published_date: #Removing jobs posted few days ago
            company_name = job.find('h3',class_ = 'joblist-comp-name').text.strip()
            skills = job.find('span',class_ = 'srp-skills').text.strip()
            more_info = job.header.h2.a['href']
            
            company_name_list.append(company_name)
            skills_list.append(skills)            
            more_info_list.append(more_info)


            if unfamiliar_skills not in skills: #Skills is a list which is being searched into

                print(f'Company Name : {company_name}')
                print(f"Skills : {skills.replace(' ','')}")
                print(f'More Info : {more_info}')
                print()

    job_data = pd.DataFrame(
    {'Company_Name': company_name_list,
     'skills': skills_list,
     'Link': more_info_list,     
    })
    job_data.to_csv('D:\Ricky\Data Science\Portfolio Projects\Web Scraping Job Portal\job_posting_details.csv', index = False)              


# In[ ]:


if __name__ == '__main__':
    while True:
        find_jobs()
        time_wait = 10
        print(f'Waiting for {time_wait} seconds')
        time.sleep(time_wait*6) #Programing code to run after every 10 seconds


# In[ ]:




