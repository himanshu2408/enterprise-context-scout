{{/*
Expand the name of the chart.
*/}}
{{- define "context-scout.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "context-scout.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "context-scout.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "context-scout.labels" -}}
helm.sh/chart: {{ include "context-scout.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: {{ include "context-scout.name" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}

{{/*
Component selector labels
*/}}
{{- define "context-scout.selectorLabels" -}}
app.kubernetes.io/name: {{ include "context-scout.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Component-specific ServiceAccount name
*/}}
{{- define "context-scout.serviceAccountName" -}}
{{- $root := index . 0 -}}
{{- $component := index . 1 -}}
{{- printf "%s-%s" (include "context-scout.fullname" $root) $component | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Workload Identity annotations for a given component (api | worker | vllm)
*/}}
{{- define "context-scout.workloadIdentityAnnotations" -}}
{{- $root := index . 0 -}}
{{- $component := index . 1 -}}
{{- $wi := $root.Values.serviceAccount.workloadIdentity -}}
{{- if and (eq $wi.provider "gcp") (index $wi.gcp $component) }}
iam.gke.io/gcp-service-account: {{ index $wi.gcp $component | quote }}
{{- else if eq $wi.provider "aws" }}
{{- if and (eq $component "api") $wi.aws.apiRoleArn }}
eks.amazonaws.com/role-arn: {{ $wi.aws.apiRoleArn | quote }}
{{- else if and (eq $component "worker") $wi.aws.workerRoleArn }}
eks.amazonaws.com/role-arn: {{ $wi.aws.workerRoleArn | quote }}
{{- else if and (eq $component "vllm") $wi.aws.vllmRoleArn }}
eks.amazonaws.com/role-arn: {{ $wi.aws.vllmRoleArn | quote }}
{{- end }}
{{- else if eq $wi.provider "azure" }}
{{- if and (eq $component "api") $wi.azure.apiClientId }}
azure.workload.identity/client-id: {{ $wi.azure.apiClientId | quote }}
{{- else if and (eq $component "worker") $wi.azure.workerClientId }}
azure.workload.identity/client-id: {{ $wi.azure.workerClientId | quote }}
{{- else if and (eq $component "vllm") $wi.azure.vllmClientId }}
azure.workload.identity/client-id: {{ $wi.azure.vllmClientId | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Cloud SQL Auth Proxy sidecar container spec (used by api and worker deployments)
*/}}
{{- define "context-scout.cloudSqlAuthProxyContainer" -}}
- name: cloud-sql-proxy
  image: {{ .Values.cloudSqlAuthProxy.image | quote }}
  imagePullPolicy: {{ .Values.global.imagePullPolicy }}
  args:
    - {{ printf "--port=%d" (int .Values.cloudSqlAuthProxy.port) | quote }}
    {{- if .Values.cloudSqlAuthProxy.autoIamAuthn }}
    - "--auto-iam-authn"
    {{- end }}
    {{- if .Values.cloudSqlAuthProxy.psc }}
    - "--psc"
    {{- else if .Values.cloudSqlAuthProxy.privateIp }}
    - "--private-ip"
    {{- end }}
    {{- if .Values.cloudSqlAuthProxy.structuredLogs }}
    - "--structured-logs"
    {{- end }}
    {{- if .Values.cloudSqlAuthProxy.healthCheck }}
    - "--health-check"
    - {{ printf "--http-port=%d" (int .Values.cloudSqlAuthProxy.httpPort) | quote }}
    {{- end }}
    - {{ .Values.cloudSqlAuthProxy.instanceConnectionName | quote }}
  ports:
    - name: sql-proxy
      containerPort: {{ int .Values.cloudSqlAuthProxy.port }}
      protocol: TCP
    - name: proxy-health
      containerPort: {{ int .Values.cloudSqlAuthProxy.httpPort }}
      protocol: TCP
  livenessProbe:
    httpGet:
      path: /liveness
      port: proxy-health
    initialDelaySeconds: 10
    periodSeconds: 10
  readinessProbe:
    httpGet:
      path: /readiness
      port: proxy-health
    initialDelaySeconds: 5
    periodSeconds: 5
  securityContext:
    {{- toYaml .Values.containerSecurityContext | nindent 4 }}
  resources:
    {{- toYaml .Values.cloudSqlAuthProxy.resources | nindent 4 }}
  volumeMounts:
    - name: tmp
      mountPath: /tmp
{{- end }}
