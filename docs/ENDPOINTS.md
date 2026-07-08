## Documentation

### Base URL
All endpoints below are served under:
```
/api/v1
```

### /asr
#### `/audio_fb`
Transcribe an uploaded audio clip (voice feedback) to text.
##### POST
`multipart/form-data`
```js
{
	audio: "[audio file: wav/mp3/mp4]"
}
```
returns
```js
{
	transcript: "I like the outfit but change the color",
	signal_type: "voice_feedback"
}
```

### /vlm
#### `/vlm-generated-clothes-captions`
Generate clothing descriptions from a background image.
##### POST
`multipart/form-data`
```js
{
	image: "[image file: png/jpg]"
}
```
returns
```js
{
	res: [
		"a light blue denim jacket",
		"white sneakers",
		"cream hoodie"
	]
}
```

#### `/vlm-clip-images-matching`
Generate descriptions from a background image, then rank clothes by PE-CLIP image similarity.
##### POST
`multipart/form-data`
```js
{
	image: "[image file: png/jpg]"
}
```
returns
```js
{
	query: ["a long beige trench coat", "black boots"],
	results: [
		{
			name_clothes: "coat_013",
			similarity: 0.7821,
			best_description: "a long beige trench coat"
		}
	]
}
```

#### `/vlm-clip-caption-matching`
Generate descriptions from a background image, then rank clothes using the captions cache.
##### POST
`multipart/form-data`
```js
{
	image: "[image file: png/jpg]"
}
```
returns
```js
{
	query: ["green bomber jacket", "black jeans"],
	results: [
		{
			name_clothes: "jacket_022",
			similarity: 0.6942,
			best_description: "green bomber jacket"
		}
	]
}
```

#### `/vlm-caption-feedback`
Rank clothes using a text feedback string and precomputed captions (no image upload required).
##### POST
`application/json`
```js
{
	descriptions: ["green bomber jacket", "black jeans"],
	fb_text: "prefer darker colors"
}
```
returns
```js
{
	query: ["green bomber jacket", "black jeans"],
	feedback: "prefer darker colors",
	results: [
		{
			name_clothes: "jacket_022",
			similarity: 0.6942,
			best_description: "green bomber jacket"
		}
	]
}
```

#### `/clothes-captions`
Return the cached captions JSON for clothes. If it doesn't exist, it is generated first.
##### GET
returns
```js
{
	"shirt_001.png": "a white oversized t-shirt",
	"pants_004.png": "slim-fit dark denim jeans"
}
```

#### `/vlm-tournament-selection`
Pick the best clothes using tournament selection over captions (batch size 10 per round).
##### POST
`multipart/form-data`
```js
{
	image: "[image file: png/jpg]"
}
```
returns
```js
{
	background_caption: "a city street at dusk",
	best_clothes: "jacket_022"
}
```
If no captions exist, `best_clothes` is null.

#### `/all-methods`
Run all three approaches in one request.
##### POST
`multipart/form-data`
```js
{
	image: "[image file: png/jpg]"
}
```
returns
```js
{
	approach_1: {
		bg_caption: "",
		query: ["beige trench coat"],
		result: {
			name_clothes: "coat_013",
			similarity: 0.7821,
			best_description: "beige trench coat"
		}
	},
	approach_2: {
		bg_caption: "",
		query: ["beige trench coat"],
		result: {
			name_clothes: "coat_013",
			similarity: 0.7014,
			best_description: "beige trench coat"
		}
	},
	approach_3: {
		bg_caption: "a city street at dusk",
		query: [],
		result: {
			name_clothes: "coat_013",
			similarity: 0,
			best_description: ""
		}
	}
}
```
