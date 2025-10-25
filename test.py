import requests

url = "https://style-transfer-api.onrender.com/transfer"
files = {
    'content': open('content.jpeg', 'rb'),
    'style': open('style.jpg', 'rb')
}
response = requests.post(url, files=files)

if response.status_code == 200:
    with open("output.jpg", "wb") as f:
        f.write(response.content)
    print("Saved stylized image as output.jpg")
else:
    print(response.text)
