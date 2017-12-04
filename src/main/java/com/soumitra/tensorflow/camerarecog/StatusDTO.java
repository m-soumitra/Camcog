package com.soumitra.tensorflow.camerarecog;

import java.io.Serializable;

public class StatusDTO implements Serializable {

	private static final long serialVersionUID = 1L;

	private String msg;
	private Integer statusCd;

	public String getMsg() {
		return msg;
	}

	public void setMsg(String msg) {
		this.msg = msg;
	}

	public Integer getStatusCd() {
		return statusCd;
	}

	public void setStatusCd(Integer statusCd) {
		this.statusCd = statusCd;
	}

}
