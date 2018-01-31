package com.soumitra.tensorflow.camerarecog;

import org.apache.tomcat.util.codec.binary.Base64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

/**
 * @author Soumitra Mohakud
 * @category This uses an already trained model to identify images.
 */
@CrossOrigin
@RestController
public class CameraRecogService {

	@Autowired
	@Qualifier("camRecog")
	private CamRecog camRecog;

	private final Logger logger = LoggerFactory.getLogger(this.getClass());

	@RequestMapping(value = "/fetchProbability", method = RequestMethod.POST)
	public StatusDTO fetchProbability(@RequestBody String encodedImage,
			@RequestParam("matchCasesReq") int matchCasesReq) {
		logger.info("Camera in Action");
		logger.info("Image Byte Array {}", encodedImage);
		byte[] imageByteData = Base64.decodeBase64(encodedImage);
		StatusDTO status = new StatusDTO();
		ClassLoader classLoader = getClass().getClassLoader();

		try {
			status.setResults(camRecog.execute(imageByteData, matchCasesReq, classLoader));
			status.setMsg("Success");
			status.setStatusCd(200);
		} catch (Exception e) {
			logger.error("Exception occurred while parsing file: ", e);
			status.setMsg(e.getMessage());
			status.setStatusCd(500);
		}
		return status;
	}
}
