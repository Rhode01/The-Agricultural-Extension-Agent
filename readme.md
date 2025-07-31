An end-to-end AI prototype that helps smallholder maize farmers diagnose leaf diseases and receive treatment advice—combining custom computer vision, a retrieval-augmented knowledge base, and an intelligent agent orchestration.

Project Overview

Smallholder farmers in Malawi often lack timely access to extension officers for diagnosing and treating maize diseases. The Agricultural Extension Agent is a prototype AI system that:

    Sees diseases in maize-leaf images.

    Understands treatment recommendations (RAG system over agricultural guides).

    Acts by orchestrating diagnosis and advice via a ReAct agent with custom tools.

Features

    Custom Object Detection

        Train YOLOv8 on annotated maize-leaf images.

    Retrieval-Augmented Generation

        Vector-store of PDF/text guides for disease treatments.

    Custom Tools

        disease_analysis_tool(image_path)

        get_treatment_advice_tool(disease_name)

    ReAct Agent

        Orchestrates the diagnosis-and-treatment workflow.