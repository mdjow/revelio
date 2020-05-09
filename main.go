package main

import (
	"encoding/base64"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/net/context"
	"golang.org/x/oauth2/google"

	vision "google.golang.org/api/vision/v1p2beta1"
)

func runLabel(file string) error {
	ctx := context.Background()

	client, err := google.DefaultClient(ctx, vision.CloudPlatformScope)
	if err != nil {
		return err
	}
	service, err := vision.New(client)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}

	req := &vision.GoogleCloudVisionV1p2beta1AnnotateImageRequest{
		Image: &vision.GoogleCloudVisionV1p2beta1Image{
			Content: base64.StdEncoding.EncodeToString(b),
		},
		Features: []*vision.GoogleCloudVisionV1p2beta1Feature{
			{
				Type:       "LABEL_DETECTION",
				MaxResults: 5,
			},
		},
	}

	batch := &vision.GoogleCloudVisionV1p2beta1BatchAnnotateImagesRequest{
		Requests: []*vision.GoogleCloudVisionV1p2beta1AnnotateImageRequest{req},
	}

	res, err := service.Images.Annotate(batch).Do()
	if err != nil {
		return err
	}

	if annotations := res.Responses[0].LabelAnnotations; len(annotations) > 0 {
		for i := 0; i < len(annotations); i++ {
			label := annotations[i].Description
			score := annotations[i].Score
			fmt.Printf("Found label: %s, Score: %f for %s\n", label, score, file)
		}
		return nil
	}
	fmt.Printf("Not found label: %s\n", file)

	return nil
}

func runText(file string) error {
	ctx := context.Background()

	client, err := google.DefaultClient(ctx, vision.CloudPlatformScope)
	if err != nil {
		return err
	}

	service, err := vision.New(client)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}

	req := &vision.GoogleCloudVisionV1p2beta1AnnotateImageRequest{
		Image: &vision.GoogleCloudVisionV1p2beta1Image{
			Content: base64.StdEncoding.EncodeToString(b),
		},
		Features: []*vision.GoogleCloudVisionV1p2beta1Feature{
			{
				Type: "TEXT_DETECTION",
			},
		},
	}

	batch := &vision.GoogleCloudVisionV1p2beta1BatchAnnotateImagesRequest{
		Requests: []*vision.GoogleCloudVisionV1p2beta1AnnotateImageRequest{req},
	}

	res, err := service.Images.Annotate(batch).Do()
	if err != nil {
		return err
	}

	if annotations := res.Responses[0].TextAnnotations; len(annotations) > 0 {
		text := annotations[0].Description
		fmt.Printf("Found text: %s\n", text)
		return nil
	}
	fmt.Printf("Not found text in: %s\n", file)

	return nil
}

func runFace(file string) error {
	ctx := context.Background()

	client, err := google.DefaultClient(ctx, vision.CloudPlatformScope)
	if err != nil {
		return err
	}
	service, err := vision.New(client)
	if err != nil {
		return err
	}

	b, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	req := &vision.GoogleCloudVisionV1p2beta1AnnotateImageRequest{
		Image: &vision.GoogleCloudVisionV1p2beta1Image{
			Content: base64.StdEncoding.EncodeToString(b),
		},
		Features: []*vision.GoogleCloudVisionV1p2beta1Feature{
			{
				Type:       "FACE_DETECTION",
				MaxResults: 5,
			},
		},
	}

	batch := &vision.GoogleCloudVisionV1p2beta1BatchAnnotateImagesRequest{
		Requests: []*vision.GoogleCloudVisionV1p2beta1AnnotateImageRequest{req},
	}

	res, err := service.Images.Annotate(batch).Do()
	if err != nil {
		return err
	}

	if annotations := res.Responses[0].FaceAnnotations; len(annotations) > 0 {
		for i, annotation := range annotations {
			face := i + 1
			anger := annotation.AngerLikelihood
			joy := annotation.JoyLikelihood
			surprise := annotation.SurpriseLikelihood

			fmt.Printf("Found Face: %d, Anger: %s Joy %s Surprise %s\n", face, anger, joy, surprise)
		}
		return nil
	}
	fmt.Printf("Not found faces: %s\n", file)

	return nil
}

type arrayFlags []string

func (out *arrayFlags) String() string {
	return ""
}

func (out *arrayFlags) Set(value string) error {
	*out = append(*out, strings.TrimSpace(value))
	return nil
}

var img arrayFlags
var revelio arrayFlags

const (
	text  = "text"
	label = "label"
	face  = "face"
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: go run %s <path-to-image> <revelio>\n", filepath.Base(os.Args[0]))
	}

	flag.Var(&img, "img", "List of images")
	flag.Var(&revelio, "revelio", "List of revelio")
	flag.Parse()

	if img == nil || revelio == nil {
		flag.Usage()
		os.Exit(1)
	}

	switch revelio[0] {
	case text:
		if err := runText(img[0]); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", err.Error())
			os.Exit(1)
		}
		break
	case label:
		if err := runLabel(img[0]); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", err.Error())
			os.Exit(1)
		}
		break
	case face:
		if err := runFace(img[0]); err != nil {
			fmt.Fprintf(os.Stderr, "%s\n", err.Error())
			os.Exit(1)
		}
		break
	default:
		fmt.Fprintf(os.Stderr, "err\n")
	}
}
