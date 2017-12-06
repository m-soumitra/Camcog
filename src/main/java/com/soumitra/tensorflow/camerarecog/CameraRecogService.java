package com.soumitra.tensorflow.camerarecog;

import java.io.File;
import java.io.InputStream;

import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

/**
 * 
 * @author Soumitra Mohakud This uses an already trained model to identify
 *         images.
 */
@CrossOrigin
@RestController
public class CameraRecogService {

	@Autowired
	@Qualifier("camRecog")
	CamRecog camRecog;

	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	@RequestMapping(value = "/fetchImageName", method = RequestMethod.GET)
	public StatusDTO cameraService() {
		logger.info("Camera in Action");
		StatusDTO status = new StatusDTO();
		ClassLoader classLoader = getClass().getClassLoader();
		String modelsPath = classLoader.getResource("models/inception5h").getPath();
		String imagePath = classLoader.getResource("Caaar.jpg").getPath();
		logger.info("Image path {} ", imagePath);
		logger.info("Model Path {} ", modelsPath);

		try (InputStream imageAsStream = classLoader.getResourceAsStream("Caaar.jpg")) {
			byte[] imageData = IOUtils.toByteArray(imageAsStream);
			status.setMsg(camRecog.execute(imageData, classLoader));
			status.setStatusCd(200);
		} catch (Exception e) {
			logger.error("Exception occurred while parsing file: ", e);
			status.setMsg(e.getMessage());
			status.setStatusCd(500);
		}
		return status;
	}

	@RequestMapping(value = "/fetchdum", method = RequestMethod.GET)
	public String cameraDummy() {
		return "test service";
	}
}
