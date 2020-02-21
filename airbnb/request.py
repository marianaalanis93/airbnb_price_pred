import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'id':10595,'name':'96m2, 3BR, 2BA, Metro, WI-FI etc...','description':'Athens Furnished Apartment No6 is 3-bedroom apartment with 2-bathrooms -excellent located  -close to metro station,  -lovely,  -very clean  with all the facilities that you will need, nice balcony, excellent Wi-Fi, cable tv, fully air conditioned‚Ä¶ Athens Furnished Apartment No6 is an excellent located, close to metro, lovely, very clean 3-bedroom apartment with 2-bathrooms with all the facilities that you will need and balcony. It is on the 2nd floor but do not worry because there is elevator in the building. Fully equipped kitchen with everything you need to prepare your lunch/dinner. Living room to relax and enjoy a movie or a sport event. 2 Clean nice bathrooms. For more than 6 people there is a sofa/bed.  Apartment No6 has everything you will need. 1st Bedroom ‚Äì Double bed 2nd Bedroom ‚Äì 2 single beds 3rd Bedroom ‚Äì 2 single beds -Telephone line for incoming calls or to call us if you need something. -Free fast Wi-Fi from the best internet provider in Greece. You do not share the con',
	'host_id':37177,'host_name':'Emmanouil','host_since':'9/8/09','host_about':'Athens Quality Apartments is a company started back at 2007 and now we have 8 apartments. Our goal is to offer to travelers beautiful apartments with professional service only in good location and with all the necessary amenities.',
	'host_response_time':'within an hour','host_is_superhost':'t','host_listings_count':6,'host_identity_verified':'t','latitude':37.98888,'longitude':23.76431,
	'room_type':'Entire home/apt','accommodates':8,'bathrooms':2,'bedrooms':3,'beds':5,'square_feet':1076,'guests_included':4,'minimum_nights':1,'maximum_nights':45,
	'availability_90':46,'number_of_reviews':22,'first_review':'5/20/11','last_review':'8/7/19','review_scores_rating':96,'review_scores_accuracy':10,
	'review_scores_cleanliness':10,'review_scores_checkin':10,'review_scores_communication':10,'review_scores_location':9,'reviews_per_month':0.21})

print(r.json())
