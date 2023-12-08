const SERVER_PATH = 'https://api-albertovalerio.cloud.okteto.net/'
const SAMPLE_PATH = SERVER_PATH + 'images'
const IMAGES_PATH = SERVER_PATH + 'temp/'
const SAMPLE_LIMIT = '9'
const PREDICT_PATH = SERVER_PATH + 'predict'
const UPLOAD_PATH = SERVER_PATH + 'upload/image'
const METRICS_PATH = SERVER_PATH + 'metrics'

const previewImgCont = document.getElementById('preview-img')
const originalImgCont = document.getElementById('original-img')
const maskImgCont = document.getElementById('predicted-img')
const truthImgCont = document.getElementById('truth-img')
const container = document.getElementById('sample-container')
const table = document.getElementById('metrics-body')
const buttonUpload = document.getElementById('upload-submit')


const toggleTab = (e) => {
	const el = e.target
	const panel = document.getElementById(el.dataset.twTarget)
	setActiveTab(el, panel)
	if (el.dataset.twTarget === 'example-tab-1') {
		container.innerHTML = ''
		showImageSample()
	}
	if (el.dataset.twTarget === 'example-tab-1' || el.dataset.twTarget === 'example-tab-2') {
		container.innerHTML = ''
		table.innerHTML = '<tr><td colspan="3"><div class="alert alert-danger-soft show flex items-center my-4 w-full" role="alert">No metrics availables!</div></td></tr>'
		setToDefault(['original', 'mask', 'preview', 'truth'])
	}
	if (el.dataset.twTarget === 'example-tab-4') {
		getMetrics()
	}
}


const setActiveTab = (btn, panel) => {
	document.querySelectorAll('#'+ btn.closest('ul').id + ' button').forEach((e) => {
		e.classList.remove('active')
		e.ariaSelected = 'false'
	})
	btn.classList.toggle('active')
	btn.ariaSelected = 'true'
	document.querySelectorAll('#'+ panel.parentNode.id + ' div[role="tabpanel"]').forEach((e) => {
		e.classList.remove('active')
	})
	panel.classList.toggle('active')
}


const selectImage = (e) => {
	let el = e.target
	const button = document.getElementById('sample-submit')
	if (el.classList.contains('my-selection')) {
		el.classList.remove('my-selection')
		button.classList.remove('btn-primary')
		button.classList.add('btn-outline-secondary', 'cursor-no-drop')
		setToDefault(['original', 'mask'])
	} else {
		document.querySelectorAll('#'+ container.id + ' img').forEach((e) => {
			e.classList.remove('my-selection')
		})
		e.target.classList.add('my-selection')
		button.classList.remove('btn-outline-secondary', 'cursor-no-drop')
		button.classList.add('btn-primary')
	}
}


const showImageSample = async () => {
	const loader = document.getElementById('sample-loader')
	container.innerHTML = ''
	loader.classList.toggle('hidden')
	await fetch(SAMPLE_PATH + '?limit=' + SAMPLE_LIMIT)
		.then(res => res.json())
		.then(data => {
			loader.classList.toggle('hidden')
			data.samples.forEach((img) => {
				container.innerHTML += '<div class="border-2 border-dashed shadow-sm border-slate-200/60 dark:border-darkmode-400 rounded-md p-5 block col-span-6 md:col-span-4"><div class="h-32 relative image-fit cursor-pointer zoom-in mx-auto"><img class="rounded-md" alt="image sample" src="'+IMAGES_PATH+img+'" onclick="selectImage(event)"></div></div>'
			})
		})
		.catch((err) => {
			showNotification(err.message)
		})
}


const submitPrediction = async (e) => {
	if (e.target.classList.contains('btn-primary')) {
		const loaders = document.querySelectorAll('.pred-loader')
		originalImgCont.classList.toggle('hidden')
		maskImgCont.classList.toggle('hidden')
		loaders.forEach((l) => l.classList.toggle('hidden'))
		setActiveTab(
			document.querySelector('#example-3-tab button'),
			document.getElementById('example-tab-3')
		)
		let originalImg = ''
		document.querySelectorAll('#sample-container img').forEach((img) => {
			if (img.classList.contains('my-selection')) {
				originalImg = img.src
			}
		})
		if (originalImg === '') {
			originalImg = previewImgCont.src
		}
		await fetch(PREDICT_PATH, {
			method: 'POST',
			headers: {"Content-Type": "application/json"},
			body: JSON.stringify({'og_name': originalImg.split('/').splice(-1)[0]})
		})
		.then(res => res.json())
		.then(data => {
			maskImgCont.src = ''
			originalImgCont.src = ''
			loaders.forEach((l) => l.classList.toggle('hidden'))
			originalImgCont.classList.toggle('hidden')
			maskImgCont.classList.toggle('hidden')
			originalImgCont.src = originalImg
			originalImgCont.classList.add('zoom-in')
			originalImgCont.classList.remove('opacity-50')
			originalImgCont.parentNode.classList.remove('image-fit')
			maskImgCont.src = IMAGES_PATH + data.mask
			maskImgCont.classList.add('zoom-in')
			maskImgCont.classList.remove('opacity-50')
			maskImgCont.parentNode.classList.remove('image-fit')
			if (window.screen.width < 1024) {
				document.getElementById('example-tab-3').scrollIntoView(
					{ behavior: "smooth", block: "end", inline: "nearest" }
				)
			}
		})
		.catch((err) => {
			showNotification(err.message)
		})
	}
}


const triggerUpload = () => {
	const input = document.getElementById('upload')
	input.click()
}


const uploadImage = async (e) => {
	const file = e.target.files[0]
	if(file && file.size <= 1000000){
		const loader = document.getElementById('upload-loader')
		loader.classList.toggle('hidden')
		previewImgCont.classList.toggle('hidden')
		setToDefault(['original','mask','preview'])
		let formData = new FormData()
		formData.append('file', file)
		await fetch(UPLOAD_PATH, {
			method: 'POST',
			body: formData
		})
		.then(res => res.json())
		.then(_ => {
			previewImgCont.src = ''
			loader.classList.toggle('hidden')
			previewImgCont.classList.toggle('hidden')
			buttonUpload.classList.add('btn-primary')
			buttonUpload.classList.remove('btn-outline-secondary', 'cursor-no-drop')
			previewImgCont.src = IMAGES_PATH + 'upload.jpg'
			previewImgCont.classList.add('zoom-in')
			previewImgCont.classList.remove('opacity-50')
			previewImgCont.parentNode.classList.remove('image-fit')
		})
		.catch((err) => {
			showNotification(err.message)
		})
	} else {
		previewImgCont.src = ''
		e.target.value = ''
		setToDefault(['original', 'mask', 'preview', 'truth'])
		if (file.size > 1000000) {
			showNotification("Attention, file too big! Max file size is 1MB!")
		}
	}
}

const setToDefault = (targets) => {
	if (targets.includes('original')) {
		originalImgCont.src = 'images/original-placeholder.jpg'
		originalImgCont.classList.remove('zoom-in')
		originalImgCont.classList.add('opacity-50')
		originalImgCont.parentNode.classList.add('image-fit')
	}
	if (targets.includes('mask')) {
		maskImgCont.src = 'images/mask-placeholder.jpg'
		maskImgCont.classList.remove('zoom-in')
		maskImgCont.classList.add('opacity-50')
		maskImgCont.parentNode.classList.add('image-fit')
	}
	if (targets.includes('preview')) {
		buttonUpload.classList.remove('btn-primary')
		buttonUpload.classList.add('btn-outline-secondary', 'cursor-no-drop')
		previewImgCont.src = 'images/preview-placeholder.jpg'
		previewImgCont.classList.remove('zoom-in')
		previewImgCont.classList.add('opacity-50')
		previewImgCont.parentNode.classList.add('image-fit')
	}
	if (targets.includes('truth')) {
		truthImgCont.src = 'images/truth-placeholder.jpg'
		truthImgCont.classList.remove('zoom-in')
		truthImgCont.classList.add('opacity-50')
		truthImgCont.parentNode.classList.add('image-fit')
	}
}


const showNotification = (msg) => {
	const notify = document.getElementById('danger-notification')
	const box = document.getElementById(notify.id + '-msg')
	notify.classList.add('on')
	box.innerHTML = msg
	setTimeout(() => {
		notify.classList.remove('on')
		box.innerHTML = ''
	}, 5000)
}


const getMetrics = async () => {
	const p = previewImgCont.src.split('/').splice(-1)[0]
	const o = originalImgCont.src.split('/').splice(-1)[0]
	const m = maskImgCont.src.split('/').splice(-1)[0]
	table.innerHTML = ''
	setToDefault(['truth'])
	if (p === 'preview-placeholder.jpg' && o !== 'original-placeholder.jpg' && m !== 'mask-placeholder.jpg') {
		const loaders = document.querySelectorAll('.metrics-loader')
		truthImgCont.classList.toggle('hidden')
		loaders.forEach((l) => l.classList.toggle('hidden'))
		await fetch(METRICS_PATH, {
			method: 'POST',
			headers: {"Content-Type": "application/json"},
			body: JSON.stringify({'mask_name': m})
		})
		.then(res => res.json())
		.then(data => {
			loaders.forEach((l) => l.classList.toggle('hidden'))
			truthImgCont.classList.toggle('hidden')
			truthImgCont.src = IMAGES_PATH + data.truth
			truthImgCont.classList.add('zoom-in')
			truthImgCont.classList.remove('opacity-50')
			truthImgCont.parentNode.classList.remove('image-fit')
			arrows = getArrows({'acc': data.acc, 'iou': data.iou})
			table.innerHTML = '<tr><td>Accuracy</td><td class="text-center">'+data.acc+'</td><td class="text-center">'+arrows.acc+'</td></tr><tr><td>Jaccard</td><td class="text-center">'+data.iou+'</td><td class="text-center">'+arrows.iou+'</td></tr>'
		})
		.catch((err) => {
			showNotification(err.message)
		})
	} else {
		table.innerHTML = '<tr><td colspan="3"><div class="alert alert-danger-soft show flex items-center my-4 w-full" role="alert">No metrics availables!</div></td></tr>'
	}
}


const getArrows = (metrics) => {
	const gold = {'acc': 76.53, 'iou': 63.38}
	const arrows = {'acc': '', 'iou': ''}

	const arrowUp = '<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 384 512" style="enable-background:new 0 0 384 512;" xml:space="preserve"><path class="arrowUp" d="M214.6,41.4c-12.5-12.5-32.8-12.5-45.3,0l-160,160c-12.5,12.5-12.5,32.8,0,45.3c12.5,12.5,32.8,12.5,45.3,0L160,141.2V448c0,17.7,14.3,32,32,32s32-14.3,32-32V141.2l105.4,105.4c12.5,12.5,32.8,12.5,45.3,0s12.5-32.8,0-45.3L214.6,41.4L214.6,41.4z"/></svg>'

	const arrowDown = '<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 384 512" style="enable-background:new 0 0 384 512;" xml:space="preserve"><path class="arrowDown" d="M169.4,470.6c12.5,12.5,32.8,12.5,45.3,0l160-160c12.5-12.5,12.5-32.8,0-45.3s-32.8-12.5-45.3,0L224,370.8V64c0-17.7-14.3-32-32-32s-32,14.3-32,32v306.7L54.6,265.4c-12.5-12.5-32.8-12.5-45.3,0s-12.5,32.8,0,45.3L169.4,470.6L169.4,470.6z"/></svg>'

	const arrowEq = '<svg version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 448 512" style="enable-background:new 0 0 448 512;" xml:space="preserve"><path class="arrowEq" d="M438.6,150.6c12.5-12.5,12.5-32.8,0-45.3l-96-96c-12.5-12.5-32.8-12.5-45.3,0s-12.5,32.8,0,45.3L338.7,96H32c-17.7,0-32,14.3-32,32s14.3,32,32,32h306.7l-41.4,41.4c-12.5,12.5-12.5,32.8,0,45.3c12.5,12.5,32.8,12.5,45.3,0L438.6,150.6L438.6,150.6z M105.3,502.6c12.5,12.5,32.8,12.5,45.3,0s12.5-32.8,0-45.3L109.3,416H416c17.7,0,32-14.3,32-32s-14.3-32-32-32H109.3l41.4-41.4c12.5-12.5,12.5-32.8,0-45.3s-32.8-12.5-45.3,0l-96,96c-12.5,12.5-12.5,32.8,0,45.3L105.3,502.6L105.3,502.6z"/></svg>'

	arrows.acc = metrics.acc < (gold.acc - 10) ? arrowDown : metrics.acc > (gold.acc + 10)? arrowUp : arrowEq
	arrows.iou = metrics.iou < (gold.iou - 10) ? arrowDown : metrics.iou > (gold.iou + 10)? arrowUp : arrowEq
	return arrows
}
