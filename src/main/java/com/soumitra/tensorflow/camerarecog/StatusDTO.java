package com.soumitra.tensorflow.camerarecog;

import java.io.Serializable;
import java.util.List;

public class StatusDTO implements Serializable {

	private static final long serialVersionUID = 1L;

	private List<String> results;
	private Integer statusCd;
	private String msg;

	public String getMsg() {
		return msg;
	}

	public void setMsg(String msg) {
		this.msg = msg;
	}

	public List<String> getResults() {
		return results;
	}

	public void setResults(List<String> results) {
		this.results = results;
	}

	public Integer getStatusCd() {
		return statusCd;
	}

	public void setStatusCd(Integer statusCd) {
		this.statusCd = statusCd;
	}

}
