package com.soumitra.tensorflow.camerarecog;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.util.ResourceUtils;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

@RestController
//@CrossOrigin(origins = { "http://localhost:8100" })
public class CameraRecogService {

	@Autowired
	@Qualifier("camRecog")
	CamRecog camRecog;

	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	@RequestMapping(value = "/fetchImageName", method = RequestMethod.GET)
	public StatusDTO cameraService() {
		logger.info("Camera in Action");
		StatusDTO status = new StatusDTO();
		File file = null;
		ClassLoader classLoader = getClass().getClassLoader();
		String modelsPath = classLoader.getResource("models/inception5h").getPath();
		String imagePath = classLoader.getResource("Caaar.jpg").getPath();
		byte[] imageData = null;

		try {
			file = ResourceUtils.getFile(imagePath);
		} catch (FileNotFoundException e) {
			logger.error("Exception occurred while reading file: ", e);
		}

		try (FileInputStream fileInputStream = new FileInputStream(file)) {
			imageData = new byte[(int) file.length()];
			fileInputStream.read(imageData);
		} catch (IOException e) {
			logger.error("Exception occurred while parsing file: ", e);
		}
		status.setMsg(camRecog.execute(modelsPath.substring(1, modelsPath.length()), imageData));
		status.setStatusCd(200);
		return status;
	}

	@RequestMapping(value = "/fetchdum", method = RequestMethod.GET)
	public String cameraDummy() {
		return "test service";
	}
}
