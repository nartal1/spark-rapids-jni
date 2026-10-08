# NVIDIA cuDF plugin JNI for Apache Spark

This repository contains native support code for the
[NVIDIA cuDF plugin for Apache Spark](https://github.com/NVIDIA/cudf-spark).

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/cudf-spark-jni)

Note: The NVIDIA cuDF plugin for Apache Spark was formerly known as the RAPIDS Accelerator for
Apache Spark.  The RAPIDS name will be sunset over time.  This repository now uses the
`cudf-spark-jni` name for its GitHub location and Maven artifact.

## Building From Source

See the [build instructions in the contributing guide](CONTRIBUTING.md#building-from-source).

## Experimental JSON-to-VARIANT integration

`VariantParser.parseJsonToVariant(strings, allowDuplicateKeys)` owns a VARIANT column
(`metadata`, `value` children) and a non-nullable UINT8 status column. Close the returned
`ParsedVariant`; its getters borrow columns. JSON null is a valid VARIANT, while SQL null,
invalid JSON, duplicate keys, size limits, and unsupported input have distinct statuses.
The caller implements Spark's strict/tolerant behavior and CPU replay; unsupported input
must not be interpreted as malformed JSON.

This prototype requires cuDF's `json-to-variant` branch at
`b659c4dd6ab4f948ba1ec9ed11f253fa35b99db0` (`parse_json_to_variant`). Build against that exact source with
`-Dcudf.path=/path/to/cudf-json-to-variant` and the corresponding libcudf build/install paths.
The JNI artifact bundles the Java classes from that source tree; verify
`cudf-java-version-info.properties` in the resulting jar before downstream testing.
The dependency must land upstream before this integration can build against normal cuDF main.

`requiresLegacySparkFallback` is a conservative Spark compatibility preflight, not JSON
validation: Unicode input/escapes and large inputs require CPU evaluation until the native
compatibility gaps are resolved. Per-row parser statuses remain mandatory even after preflight.
`requiresLegacyMetadataFallback` protects native extraction after CPU replay: Spark may sort a
surrogate key before replacing it with '?' in the metadata dictionary, invalidating native binary
search even for ASCII paths. It checks dictionary text rather than binary headers/offsets;
ordinary wide ASCII objects and valid Unicode siblings remain eligible.
