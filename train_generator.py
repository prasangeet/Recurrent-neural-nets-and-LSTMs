# Generate a balanced dataset: 500 male + 500 female Indian names (no numbers, natural variants)

import random

male_base = [
"Aarav","Aditya","Arjun","Aryan","Ayaan","Dev","Dhruv","Harsh","Ishaan","Kabir","Karan","Kartik",
"Kunal","Lakshya","Manav","Manish","Mohit","Nakul","Naman","Nikhil","Nirav","Om","Parth","Pranav",
"Prateek","Rahul","Raj","Rajat","Rakesh","Raman","Ramesh","Rishi","Ritesh","Rohan","Rohit",
"Rudra","Sachin","Sagar","Sahil","Samar","Sameer","Sanjay","Sanket","Shaurya","Shiv","Shivam",
"Shreyas","Siddharth","Soham","Sourabh","Subhash","Sudhir","Sumit","Suraj","Tarun","Tejas",
"Tushar","Uday","Utkarsh","Vaibhav","Varun","Vedant","Veer","Vijay","Vikas","Vikram","Vinay",
"Vineet","Vipul","Viraj","Vishal","Vishnu","Vivek","Yash","Yatin","Yogesh","Yuvraj"
]

female_base = [
"Aarohi","Aarti","Aditi","Aishwarya","Akanksha","Akshara","Amrita","Ananya","Anika","Anita",
"Anjali","Ankita","Anusha","Anushka","Arpita","Avantika","Bhavana","Chandni","Charu","Deepa",
"Deepika","Divya","Esha","Gauri","Gayatri","Harini","Harsha","Ira","Ishita","Jahnavi",
"Janvi","Jaya","Jyoti","Kajal","Kalyani","Kamini","Kanika","Karishma","Kavita","Khushi",
"Kiran","Komal","Kritika","Lata","Lavanya","Madhuri","Mahima","Malini","Meera","Megha",
"Monika","Naina","Namrata","Nandini","Neha","Nikita","Nisha","Pallavi","Pooja","Prachi",
"Pragya","Pratibha","Preeti","Priya","Priyanka","Rachana","Radha","Radhika","Raksha","Rani",
"Rashmi","Ritu","Riya","Sakshi","Saloni","Sanjana","Sarika","Shalini","Sharmila","Shilpa",
"Shivani","Shraddha","Shreya","Sneha","Sonali","Sonal","Sonia","Suhani","Swati","Tanvi",
"Trisha","Urvashi","Vaishali","Vandana","Vidya","Yamini"
]

male_suffix = ["ansh","raj","deep","veer","endra","it","esh","vansh","jeet","pal"]
female_suffix = ["ika","ita","ya","isha","angi","lata","mala","shree","priya","anshi"]

def expand_names(base_list, suffixes, target):
    names = set(base_list)
    while len(names) < target:
        base = random.choice(base_list)
        suf = random.choice(suffixes)
        new = base + suf
        new = new.capitalize()
        names.add(new)
    return list(names)[:target]

male_names = expand_names(male_base, male_suffix, 500)
female_names = expand_names(female_base, female_suffix, 500)

all_names = male_names + female_names
random.shuffle(all_names)

path = "TrainingNames.txt"
with open(path, "w") as f:
    for n in all_names:
        f.write(n + "\n")

