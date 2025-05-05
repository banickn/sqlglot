from __future__ import annotations

import typing as t

from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
    Dialect,
    NormalizationStrategy,
    binary_from_function,
    bool_xor_sql,
    # date_trunc_to_time,
    datestrtodate_sql,
    encode_decode_sql,
    build_formatted_time,
    if_sql,
    left_to_substring_sql,
    no_ilike_sql,
    no_pivot_sql,
    no_timestamp_sql,
    regexp_extract_sql,
    rename_func,
    right_to_substring_sql,
    sha256_sql,
    strposition_sql,
    struct_extract_sql,
    # timestamptrunc_sql,
    timestrtotime_sql,
    ts_or_ds_add_cast,
    unit_to_str,
    sequence_sql,
    build_regexp_extract,
)
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
from sqlglot.transforms import unqualify_columns
from sqlglot.generator import unsupported_args

DATE_ADD_OR_SUB = t.Union[exp.DateAdd, exp.TimestampAdd, exp.DateSub]


def _initcap_sql(self: Denodo.Generator, expression: exp.Initcap) -> str:
    regex = r"(\w)(\w*)"
    return f"REGEXP_REPLACE({self.sql(expression, 'this')}, '{regex}', x -> UPPER(x[1]) || LOWER(x[2]))"


def _no_sort_array(self: Denodo.Generator, expression: exp.SortArray) -> str:
    if expression.args.get("asc") == exp.false():
        comparator = "(a, b) -> CASE WHEN a < b THEN 1 WHEN a > b THEN -1 ELSE 0 END"
    else:
        comparator = None
    return self.func("ARRAY_SORT", expression.this, comparator)


def _schema_sql(self: Denodo.Generator, expression: exp.Schema) -> str:
    if isinstance(expression.parent, exp.Property):
        columns = ", ".join(f"'{c.name}'" for c in expression.expressions)
        return f"ARRAY[{columns}]"

    if expression.parent:
        for schema in expression.parent.find_all(exp.Schema):
            column_defs = schema.find_all(exp.ColumnDef)
            if column_defs and isinstance(schema.parent, exp.Property):
                expression.expressions.extend(column_defs)

    return self.schema_sql(expression)


def _quantile_sql(self: Denodo.Generator, expression: exp.Quantile) -> str:
    self.unsupported("Denodo does not support exact quantiles")
    return self.func(
        "APPROX_PERCENTILE", expression.this, expression.args.get("quantile")
    )


def _str_to_time_sql(
    self: Denodo.Generator, expression: exp.StrToDate | exp.StrToTime | exp.TsOrDsToDate
) -> str:
    return self.func("DATE_PARSE", expression.this, self.format_time(expression))


def _ts_or_ds_to_date_sql(self: Denodo.Generator, expression: exp.TsOrDsToDate) -> str:
    time_format = self.format_time(expression)
    if time_format and time_format not in (Denodo.TIME_FORMAT, Denodo.DATE_FORMAT):
        return self.sql(
            exp.cast(_str_to_time_sql(self, expression), exp.DataType.Type.DATE)
        )
    return self.sql(
        exp.cast(
            exp.cast(expression.this, exp.DataType.Type.TIMESTAMP),
            exp.DataType.Type.DATE,
        )
    )


def _ts_or_ds_add_sql(self: Denodo.Generator, expression: exp.TsOrDsAdd) -> str:
    expression = ts_or_ds_add_cast(expression)
    unit = unit_to_str(expression)
    return self.func("DATE_ADD", unit, expression.expression, expression.this)


def _ts_or_ds_diff_sql(self: Denodo.Generator, expression: exp.TsOrDsDiff) -> str:
    this = exp.cast(expression.this, exp.DataType.Type.TIMESTAMP)
    expr = exp.cast(expression.expression, exp.DataType.Type.TIMESTAMP)
    unit = unit_to_str(expression)
    return self.func("DATE_DIFF", unit, expr, this)


def _build_approx_percentile(args: t.List) -> exp.Expression:
    if len(args) == 4:
        return exp.ApproxQuantile(
            this=seq_get(args, 0),
            weight=seq_get(args, 1),
            quantile=seq_get(args, 2),
            accuracy=seq_get(args, 3),
        )
    if len(args) == 3:
        return exp.ApproxQuantile(
            this=seq_get(args, 0), quantile=seq_get(args, 1), accuracy=seq_get(args, 2)
        )
    return exp.ApproxQuantile.from_arg_list(args)


def _build_from_unixtime(args: t.List) -> exp.Expression:
    if len(args) == 3:
        return exp.UnixToTime(
            this=seq_get(args, 0),
            hours=seq_get(args, 1),
            minutes=seq_get(args, 2),
        )
    if len(args) == 2:
        return exp.UnixToTime(this=seq_get(args, 0), zone=seq_get(args, 1))

    return exp.UnixToTime.from_arg_list(args)


def _first_last_sql(self: Denodo.Generator, expression: exp.Func) -> str:
    """
    Trino doesn't support FIRST / LAST as functions, but they're valid in the context
    of MATCH_RECOGNIZE, so we need to preserve them in that case. In all other cases
    they're converted into an ARBITRARY call.

    Reference: https://trino.io/docs/current/sql/match-recognize.html#logical-navigation-functions
    """
    if isinstance(
        expression.find_ancestor(exp.MatchRecognize, exp.Select), exp.MatchRecognize
    ):
        return self.function_fallback_sql(expression)

    return rename_func("ARBITRARY")(self, expression)


def _unix_to_time_sql(self: Denodo.Generator, expression: exp.UnixToTime) -> str:
    scale = expression.args.get("scale")
    timestamp = self.sql(expression, "this")
    if scale in (None, exp.UnixToTime.SECONDS):
        return rename_func("FROM_UNIXTIME")(self, expression)

    return f"FROM_UNIXTIME(CAST({timestamp} AS DOUBLE) / POW(10, {scale}))"


def _to_int(self: Denodo.Generator, expression: exp.Expression) -> exp.Expression:
    if not expression.type:
        from sqlglot.optimizer.annotate_types import annotate_types

        annotate_types(expression, dialect=self.dialect)
    if expression.type and expression.type.this not in exp.DataType.INTEGER_TYPES:
        return exp.cast(expression, to=exp.DataType.Type.BIGINT)
    return expression


def _build_to_char(args: t.List) -> exp.TimeToStr:
    fmt = seq_get(args, 1)
    if isinstance(fmt, exp.Literal):
        # We uppercase this to match Teradata's format mapping keys
        fmt.set("this", fmt.this.upper())

    return build_formatted_time(exp.TimeToStr, "teradata")(args)


def _date_add_sql(
    kind: str,
) -> t.Callable[[generator.Generator, exp.Expression], str]:
    def func(self: generator.Generator, expression: exp.Expression) -> str:
        print("asdkajlsdlaksd")
        return self.func(
            f"ADDDAY",
            expression.this,
            exp.Interval(this=expression.expression, unit=expression),
        )

    return func


def _date_delta_sql(
    name: str, negate_interval: bool = False
) -> t.Callable[[Denodo.Generator, DATE_ADD_OR_SUB], str]:
    def _delta_sql(self: Denodo.Generator, expression: DATE_ADD_OR_SUB) -> str:
        interval = _to_int(self, expression.expression)
        print("lol")
        return self.func(
            name,
            unit_to_str(expression),
            interval * (-1) if negate_interval else interval,
            expression.this,
        )

    return _delta_sql


def _build_pad_parser(args: t.List, is_left: bool) -> exp.Pad:
    """Creates an exp.Pad expression from parsed arguments."""
    # seq_get gracefully handles missing optional arguments like fill_pattern
    return exp.Pad(
        this=seq_get(args, 0),
        expression=seq_get(args, 1),  # This is the length argument
        fill_pattern=seq_get(args, 2),  # Optional fill pattern
        # Set the crucial is_left flag in the expression's args dictionary
        args={"is_left": is_left},
    )


# This function now explicitly handles exp.Pad
def _pad_sql(self: Denodo.Generator, expression: exp.Pad) -> str:
    """
    Generates SQL for LPAD or RPAD functions from an exp.Pad expression.

    Denodo typically uses LPAD and RPAD with three arguments:
    LPAD(string, length, fill_pattern)
    RPAD(string, length, fill_pattern)

    SQLGlot's exp.Pad stores:
        - string in 'this'
        - length in 'expression'
        - fill_pattern in 'fill_pattern' (within args)
        - direction in 'is_left' (within args)
    """
    this = self.sql(expression, "this")
    length = self.sql(expression, "expression")
    # Default fill pattern to a single space if not provided
    fill_pattern = self.sql(expression.args.get("fill_pattern")) or "' '"

    # Determine the correct Denodo function name based on the is_left flag
    is_left = expression.args.get("is_left", True)  # Default to LPAD if flag missing
    func_name = "LPAD" if is_left else "RPAD"
    # If Denodo uses different names (e.g., LEFTPAD), adjust here:
    # func_name = "LEFTPAD" if is_left else "RPAD"

    # Generate the function call using the determined name and arguments
    return self.func(func_name, this, length, fill_pattern)


class Denodo(Dialect):
    INDEX_OFFSET = 1
    NULL_ORDERING = "nulls_are_last"
    TIME_FORMAT = MySQL.TIME_FORMAT
    STRICT_STRING_CONCAT = True
    SUPPORTS_SEMI_ANTI_JOIN = False
    TYPED_DIVISION = True
    TABLESAMPLE_SIZE_IS_PERCENT = True
    LOG_BASE_FIRST: t.Optional[bool] = None
    SUPPORTS_VALUES_DEFAULT = False

    TIME_MAPPING = MySQL.TIME_MAPPING

    NORMALIZATION_STRATEGY = NormalizationStrategy.CASE_INSENSITIVE

    ANNOTATORS = {
        **Dialect.ANNOTATORS,
        exp.Floor: lambda self, e: self._annotate_by_args(e, "this"),
        exp.Ceil: lambda self, e: self._annotate_by_args(e, "this"),
        exp.Mod: lambda self, e: self._annotate_by_args(e, "this", "expression"),
        exp.Round: lambda self, e: self._annotate_by_args(e, "this"),
        exp.Sign: lambda self, e: self._annotate_by_args(e, "this"),
        exp.Abs: lambda self, e: self._annotate_by_args(e, "this"),
        exp.Rand: lambda self, e: self._annotate_by_args(e, "this")
        if e.this
        else self._set_type(e, exp.DataType.Type.DOUBLE),
    }

    class Tokenizer(tokens.Tokenizer):
        UNICODE_STRINGS = [
            (prefix + q, q)
            for q in t.cast(t.List[str], tokens.Tokenizer.QUOTES)
            for prefix in ("U&", "u&")
        ]

        KEYWORDS = {
            **tokens.Tokenizer.KEYWORDS,
            "DEALLOCATE PREPARE": TokenType.COMMAND,
            "DESCRIBE INPUT": TokenType.COMMAND,
            "DESCRIBE OUTPUT": TokenType.COMMAND,
            "RESET SESSION": TokenType.COMMAND,
            "START": TokenType.BEGIN,
            "MATCH_RECOGNIZE": TokenType.MATCH_RECOGNIZE,
            "ROW": TokenType.STRUCT,
            "IPADDRESS": TokenType.IPADDRESS,
            "IPPREFIX": TokenType.IPPREFIX,
            "TDIGEST": TokenType.TDIGEST,
            "HYPERLOGLOG": TokenType.HLLSKETCH,
        }
        KEYWORDS.pop("/*+")
        KEYWORDS.pop("QUALIFY")

    class Parser(parser.Parser):
        VALUES_FOLLOWED_BY_PAREN = False

        FUNCTIONS = {
            **parser.Parser.FUNCTIONS,
            "ARBITRARY": exp.AnyValue.from_arg_list,
            "APPROX_DISTINCT": exp.ApproxDistinct.from_arg_list,
            "APPROX_PERCENTILE": _build_approx_percentile,
            "BITWISE_AND": binary_from_function(exp.BitwiseAnd),
            "BITWISE_NOT": lambda args: exp.BitwiseNot(this=seq_get(args, 0)),
            "BITWISE_OR": binary_from_function(exp.BitwiseOr),
            "BITWISE_XOR": binary_from_function(exp.BitwiseXor),
            "CARDINALITY": exp.ArraySize.from_arg_list,
            "CONTAINS": exp.ArrayContains.from_arg_list,
            "ADDDAY": lambda args: exp.DateAdd(
                this=seq_get(args, 2),
                expression=seq_get(args, 1),
                unit=seq_get(args, 0),
            ),
            "ADDHOUR": lambda args: exp.TimestampAdd(
                this=seq_get(
                    args, 0
                ),  # First arg is the value (time, timestamp, interval)
                expression=seq_get(args, 1),  # Second arg is the increment (hours)
                unit=exp.Literal.string("HOUR"),  # Explicitly set the unit
            ),
            "DATE_DIFF": lambda args: exp.DateDiff(
                this=seq_get(args, 2),
                expression=seq_get(args, 1),
                unit=seq_get(args, 0),
            ),
            "FORMATDATE": build_formatted_time(exp.TimeToStr, "denodo"),
            "DATE_PARSE": build_formatted_time(exp.StrToTime, "denodo"),
            "DAY_OF_WEEK": exp.DayOfWeekIso.from_arg_list,
            "ELEMENT_AT": lambda args: exp.Bracket(
                this=seq_get(args, 0),
                expressions=[seq_get(args, 1)],
                offset=1,
                safe=True,
            ),
            "FROM_HEX": exp.Unhex.from_arg_list,
            "FROM_UNIXTIME": _build_from_unixtime,
            "FROM_UTF8": lambda args: exp.Decode(
                this=seq_get(args, 0),
                replace=seq_get(args, 1),
                charset=exp.Literal.string("utf-8"),
            ),
            "LEVENSHTEIN_DISTANCE": exp.Levenshtein.from_arg_list,
            "NOW": exp.CurrentTimestamp.from_arg_list,
            "LPAD": lambda args: _build_pad_parser(args, is_left=True),
            "RPAD": lambda args: _build_pad_parser(args, is_left=False),
            "REGEXP_EXTRACT": build_regexp_extract(exp.RegexpExtract),
            "REGEXP_EXTRACT_ALL": build_regexp_extract(exp.RegexpExtractAll),
            "REGEXP": lambda args: exp.RegexpReplace(
                this=seq_get(args, 0),
                expression=seq_get(args, 1),
                replacement=seq_get(args, 2) or exp.Literal.string(""),
            ),
            "ROW": exp.Struct.from_arg_list,
            "SEQUENCE": exp.GenerateSeries.from_arg_list,
            "SET_AGG": exp.ArrayUniqueAgg.from_arg_list,
            "SPLIT_TO_MAP": exp.StrToMap.from_arg_list,
            "STRPOS": lambda args: exp.StrPosition(
                this=seq_get(args, 0),
                substr=seq_get(args, 1),
                occurrence=seq_get(args, 2),
            ),
            "TO_CHAR": _build_to_char,
            "TO_UNIXTIME": exp.TimeToUnix.from_arg_list,
            "TO_UTF8": lambda args: exp.Encode(
                this=seq_get(args, 0), charset=exp.Literal.string("utf-8")
            ),
            "MD5": exp.MD5Digest.from_arg_list,
            "SHA256": lambda args: exp.SHA2(
                this=seq_get(args, 0), length=exp.Literal.number(256)
            ),
            "SHA512": lambda args: exp.SHA2(
                this=seq_get(args, 0), length=exp.Literal.number(512)
            ),
        }

        FUNCTION_PARSERS = parser.Parser.FUNCTION_PARSERS.copy()
        FUNCTION_PARSERS.pop("TRIM")

    class Generator(generator.Generator):
        INTERVAL_ALLOWS_PLURAL_FORM = False
        JOIN_HINTS = False
        TABLE_HINTS = False
        QUERY_HINTS = False
        IS_BOOL_ALLOWED = False
        TZ_TO_WITH_TIME_ZONE = True
        NVL2_SUPPORTED = False
        STRUCT_DELIMITER = ("(", ")")
        LIMIT_ONLY_LITERALS = True
        SUPPORTS_SINGLE_ARG_CONCAT = False
        LIKE_PROPERTY_INSIDE_SCHEMA = True
        MULTI_ARG_DISTINCT = False
        SUPPORTS_TO_NUMBER = False
        HEX_FUNC = "TO_HEX"
        LPAD = "LPAD"
        PARSE_JSON_NAME = "JSON_PARSE"
        PAD_FILL_PATTERN_IS_REQUIRED = True
        EXCEPT_INTERSECT_SUPPORT_ALL_CLAUSE = False
        SUPPORTS_MEDIAN = False
        ARRAY_SIZE_NAME = "CARDINALITY"

        PROPERTIES_LOCATION = {
            **generator.Generator.PROPERTIES_LOCATION,
            exp.LocationProperty: exp.Properties.Location.UNSUPPORTED,
            exp.VolatileProperty: exp.Properties.Location.UNSUPPORTED,
        }

        TYPE_MAPPING = {
            **generator.Generator.TYPE_MAPPING,
            # exp.DataType.Type.BINARY: "VARBINARY",
            exp.DataType.Type.BIT: "BOOLEAN",
            exp.DataType.Type.DATETIME: "TIMESTAMP",
            exp.DataType.Type.DATETIME64: "TIMESTAMP",
            exp.DataType.Type.FLOAT: "FLOAT",
            exp.DataType.Type.HLLSKETCH: "HYPERLOGLOG",
            exp.DataType.Type.INT: "INT",
            exp.DataType.Type.STRUCT: "ROW",
            exp.DataType.Type.TEXT: "TEXT",
            exp.DataType.Type.TIMESTAMPTZ: "TIMESTAMPTZ",
            exp.DataType.Type.TIMETZ: "TIME",
        }

        TRANSFORMS = {
            **generator.Generator.TRANSFORMS,
            exp.AnyValue: rename_func("ARBITRARY"),
            exp.ApproxQuantile: rename_func("APPROX_PERCENTILE"),
            exp.ArgMax: rename_func("MAX_BY"),
            exp.ArgMin: rename_func("MIN_BY"),
            exp.Array: lambda self, e: f"ARRAY[{self.expressions(e, flat=True)}]",
            exp.ArrayAny: rename_func("ANY_MATCH"),
            exp.ArrayConcat: rename_func("CONCAT"),
            exp.ArrayContains: rename_func("CONTAINS"),
            exp.ArrayToString: rename_func("ARRAY_JOIN"),
            exp.ArrayUniqueAgg: rename_func("SET_AGG"),
            exp.AtTimeZone: rename_func("AT_TIMEZONE"),
            exp.BitwiseAnd: lambda self, e: self.func(
                "BITWISE_AND", e.this, e.expression
            ),
            exp.BitwiseLeftShift: lambda self, e: self.func(
                "BITWISE_ARITHMETIC_SHIFT_LEFT", e.this, e.expression
            ),
            exp.BitwiseNot: lambda self, e: self.func("BITWISE_NOT", e.this),
            exp.BitwiseOr: lambda self, e: self.func(
                "BITWISE_OR", e.this, e.expression
            ),
            exp.BitwiseRightShift: lambda self, e: self.func(
                "BITWISE_ARITHMETIC_SHIFT_RIGHT", e.this, e.expression
            ),
            exp.BitwiseXor: lambda self, e: self.func(
                "BITWISE_XOR", e.this, e.expression
            ),
            exp.Cast: transforms.preprocess([transforms.epoch_cast_to_ts]),
            exp.CurrentTime: lambda *_: "CURRENT_TIMESTAMP",
            exp.CurrentTimestamp: lambda *_: "CURRENT_TIMESTAMP",
            exp.CurrentUser: lambda *_: "CURRENT_USER",
            exp.DateSub: _date_add_sql("SUB"),
            exp.DateDiff: lambda self, e: self.func(
                "DATE_DIFF", unit_to_str(e), e.expression, e.this
            ),
            exp.DateStrToDate: datestrtodate_sql,
            exp.DateToDi: lambda self,
            e: f"CAST(DATE_FORMAT({self.sql(e, 'this')}, {Denodo.DATEINT_FORMAT}) AS INT)",
            exp.DateAdd: _date_add_sql("ADD"),
            exp.DayOfWeek: rename_func("GETDAYOFWEEK"),
            exp.DayOfYear: rename_func("GETDAYOFYEAR"),
            exp.DayOfWeekIso: rename_func("DAY_OF_WEEK"),
            exp.Decode: lambda self, e: encode_decode_sql(self, e, "FROM_UTF8"),
            exp.DiToDate: lambda self,
            e: f"CAST(DATE_PARSE(CAST({self.sql(e, 'this')} AS VARCHAR), {Denodo.DATEINT_FORMAT}) AS DATE)",
            exp.Encode: lambda self, e: encode_decode_sql(self, e, "TO_UTF8"),
            exp.FileFormatProperty: lambda self, e: f"FORMAT='{e.name.upper()}'",
            exp.First: _first_last_sql,
            exp.FromTimeZone: lambda self,
            e: f"WITH_TIMEZONE({self.sql(e, 'this')}, {self.sql(e, 'zone')}) AT TIME ZONE 'UTC'",
            exp.GenerateSeries: sequence_sql,
            exp.GenerateDateArray: sequence_sql,
            exp.Group: transforms.preprocess([transforms.unalias_group]),
            exp.If: if_sql(),
            exp.ILike: no_ilike_sql,
            exp.Initcap: _initcap_sql,
            exp.JSONExtract: lambda self, e: self.jsonextract_sql(e),
            exp.Last: _first_last_sql,
            exp.LastDay: lambda self, e: self.func("LAST_DAY_OF_MONTH", e.this),
            exp.Left: left_to_substring_sql,
            exp.Pad: _pad_sql,  # Handle the generic exp.Pad expression
            exp.Levenshtein: unsupported_args(
                "ins_cost", "del_cost", "sub_cost", "max_dist"
            )(rename_func("LEVENSHTEIN_DISTANCE")),
            exp.Length: rename_func("LEN"),
            exp.LogicalAnd: rename_func("BOOL_AND"),
            exp.LogicalOr: rename_func("BOOL_OR"),
            # exp.Hex: lambda self, e: self.func("UPPER", self.func("TO_HEX", self.sql(e, "this"))),
            exp.Pad: _pad_sql,
            exp.Pivot: no_pivot_sql,
            exp.Quantile: _quantile_sql,
            exp.RegexpExtract: regexp_extract_sql,
            exp.RegexpExtractAll: regexp_extract_sql,
            exp.Right: right_to_substring_sql,
            exp.Schema: _schema_sql,
            exp.SchemaCommentProperty: lambda self, e: self.naked_property(e),
            exp.Select: transforms.preprocess(
                [
                    transforms.eliminate_semi_and_anti_joins,
                    transforms.eliminate_qualify,
                ]
            ),
            exp.SortArray: _no_sort_array,
            exp.StrPosition: lambda self, e: strposition_sql(
                self, e, supports_occurrence=True
            ),
            exp.StrToDate: lambda self, e: f"CAST({_str_to_time_sql(self, e)} AS DATE)",
            exp.StrToMap: rename_func("SPLIT_TO_MAP"),
            exp.StrToTime: _str_to_time_sql,
            exp.StructExtract: struct_extract_sql,
            exp.Table: transforms.preprocess([transforms.unnest_generate_series]),
            exp.Timestamp: no_timestamp_sql,
            exp.TimestampAdd: _date_delta_sql("DATE_ADD"),
            exp.TimeStrToDate: timestrtotime_sql,
            exp.TimeStrToTime: timestrtotime_sql,
            exp.TimeStrToUnix: lambda self, e: self.func(
                "TO_UNIXTIME", self.func("DATE_PARSE", e.this, Denodo.TIME_FORMAT)
            ),
            exp.TimeToStr: lambda self, e: self.func(
                "FORMATDATE", self.format_time(e), e.this
            ),
            exp.TimeToUnix: rename_func("TO_UNIXTIME"),
            exp.ToChar: lambda self, e: self.func(
                "FORMATDATE", self.format_time(e), e.this
            ),
            exp.TryCast: transforms.preprocess([transforms.epoch_cast_to_ts]),
            exp.TsOrDiToDi: lambda self,
            e: f"CAST(SUBSTR(REPLACE(CAST({self.sql(e, 'this')} AS VARCHAR), '-', ''), 1, 8) AS INT)",
            exp.TsOrDsAdd: _ts_or_ds_add_sql,
            exp.TsOrDsDiff: _ts_or_ds_diff_sql,
            exp.TsOrDsToDate: _ts_or_ds_to_date_sql,
            exp.Unhex: rename_func("FROM_HEX"),
            exp.UnixToStr: lambda self,
            e: f"DATE_FORMAT(FROM_UNIXTIME({self.sql(e, 'this')}), {self.format_time(e)})",
            exp.UnixToTime: _unix_to_time_sql,
            exp.UnixToTimeStr: lambda self,
            e: f"CAST(FROM_UNIXTIME({self.sql(e, 'this')}) AS VARCHAR)",
            exp.VariancePop: rename_func("VAR_POP"),
            exp.WeekOfYear: rename_func("GETWEEK"),
            exp.With: transforms.preprocess(
                [transforms.add_recursive_cte_column_names]
            ),
            exp.WithinGroup: transforms.preprocess(
                [transforms.remove_within_group_for_percentiles]
            ),
            exp.Xor: bool_xor_sql,
            exp.MD5Digest: rename_func("MD5"),
            exp.SHA: rename_func("SHA1"),
            exp.SHA2: sha256_sql,
        }

        RESERVED_KEYWORDS = {
            "alter",
            "and",
            "as",
            "between",
            "by",
            "case",
            "cast",
            "constraint",
            "create",
            "cross",
            "current_time",
            "current_timestamp",
            "deallocate",
            "delete",
            "describe",
            "distinct",
            "drop",
            "else",
            "end",
            "escape",
            "except",
            "execute",
            "exists",
            "extract",
            "false",
            "for",
            "from",
            "full",
            "group",
            "having",
            "in",
            "inner",
            "insert",
            "intersect",
            "into",
            "is",
            "join",
            "left",
            "like",
            "natural",
            "not",
            "null",
            "on",
            "or",
            "order",
            "outer",
            "prepare",
            "right",
            "select",
            "table",
            "then",
            "true",
            "union",
            "using",
            "values",
            "when",
            "where",
            "with",
        }

        def md5_sql(self, expression: exp.MD5) -> str:
            this = expression.this

            if not this.type:
                from sqlglot.optimizer.annotate_types import annotate_types

                this = annotate_types(this)

            if this.is_type(*exp.DataType.TEXT_TYPES):
                this = exp.Encode(this=this, charset=exp.Literal.string("utf-8"))

            return self.func(
                "LOWER", self.func("TO_HEX", self.func("MD5", self.sql(this)))
            )

        def strtounix_sql(self, expression: exp.StrToUnix) -> str:
            # Since `TO_UNIXTIME` requires a `TIMESTAMP`, we need to parse the argument into one.
            # To do this, we first try to `DATE_PARSE` it, but since this can fail when there's a
            # timezone involved, we wrap it in a `TRY` call and use `PARSE_DATETIME` as a fallback,
            # which seems to be using the same time mapping as Hive, as per:
            # https://joda-time.sourceforge.net/apidocs/org/joda/time/format/DateTimeFormat.html
            this = expression.this
            value_as_text = exp.cast(this, exp.DataType.Type.TEXT)
            value_as_timestamp = (
                exp.cast(this, exp.DataType.Type.TIMESTAMP) if this.is_string else this
            )

            parse_without_tz = self.func(
                "DATE_PARSE", value_as_text, self.format_time(expression)
            )

            formatted_value = self.func(
                "DATE_FORMAT", value_as_timestamp, self.format_time(expression)
            )
            parse_with_tz = self.func(
                "PARSE_DATETIME",
                formatted_value,
                self.format_time(
                    expression, Hive.INVERSE_TIME_MAPPING, Hive.INVERSE_TIME_TRIE
                ),
            )
            coalesced = self.func(
                "COALESCE", self.func("TRY", parse_without_tz), parse_with_tz
            )
            return self.func("TO_UNIXTIME", coalesced)

        def bracket_sql(self, expression: exp.Bracket) -> str:
            if expression.args.get("safe"):
                return self.func(
                    "ELEMENT_AT",
                    expression.this,
                    seq_get(
                        apply_index_offset(
                            expression.this,
                            expression.expressions,
                            1 - expression.args.get("offset", 0),
                        ),
                        0,
                    ),
                )
            return super().bracket_sql(expression)

        def struct_sql(self, expression: exp.Struct) -> str:
            from sqlglot.optimizer.annotate_types import annotate_types

            expression = annotate_types(expression)
            values: t.List[str] = []
            schema: t.List[str] = []
            unknown_type = False

            for e in expression.expressions:
                if isinstance(e, exp.PropertyEQ):
                    if e.type and e.type.is_type(exp.DataType.Type.UNKNOWN):
                        unknown_type = True
                    else:
                        schema.append(f"{self.sql(e, 'this')} {self.sql(e.type)}")
                    values.append(self.sql(e, "expression"))
                else:
                    values.append(self.sql(e))

            size = len(expression.expressions)

            if not size or len(schema) != size:
                if unknown_type:
                    self.unsupported(
                        "Cannot convert untyped key-value definitions (try annotate_types)."
                    )
                return self.func("ROW", *values)
            return f"CAST(ROW({', '.join(values)}) AS ROW({', '.join(schema)}))"

        def interval_sql(self, expression: exp.Interval) -> str:
            if expression.this and expression.text("unit").upper().startswith("WEEK"):
                return f"({expression.this.name} * INTERVAL '7' DAY)"
            return super().interval_sql(expression)

        def offset_limit_modifiers(
            self,
            expression: exp.Expression,
            fetch: bool,
            limit: t.Optional[exp.Fetch | exp.Limit],
        ) -> t.List[str]:
            return [
                self.sql(expression, "offset"),
                self.sql(limit),
            ]

        def delete_sql(self, expression: exp.Delete) -> str:
            """
            Denodo only supports DELETE FROM for a single table without an alias, so we need
            to remove the unnecessary parts. If the original DELETE statement contains more
            than one table to be deleted, we can't safely map it 1-1 to a Denodo statement.
            """
            tables = expression.args.get("tables") or [expression.this]
            if len(tables) > 1:
                return super().delete_sql(expression)

            table = tables[0]
            expression.set("this", table)
            expression.set("tables", None)

            if isinstance(table, exp.Table):
                table_alias = table.args.get("alias")
                if table_alias:
                    table_alias.pop()
                    expression = t.cast(
                        exp.Delete, expression.transform(unqualify_columns)
                    )

            return super().delete_sql(expression)

        def jsonextract_sql(self, expression: exp.JSONExtract) -> str:
            is_json_extract = self.dialect.settings.get(
                "variant_extract_is_json_extract", True
            )

            # Generate JSON_EXTRACT unless the user has configured that a Snowflake / Databricks
            # VARIANT extract (e.g. col:x.y) should map to dot notation (i.e ROW access) in Denodo/Trino
            if not expression.args.get("variant_extract") or is_json_extract:
                return self.func(
                    "JSON_EXTRACT",
                    expression.this,
                    expression.expression,
                    *expression.expressions,
                )

            this = self.sql(expression, "this")

            # Convert the JSONPath extraction `JSON_EXTRACT(col, '$.x.y) to a ROW access col.x.y
            segments = []
            for path_key in expression.expression.expressions[1:]:
                if not isinstance(path_key, exp.JSONPathKey):
                    # Cannot transpile subscripts, wildcards etc to dot notation
                    self.unsupported(
                        f"Cannot transpile JSONPath segment '{path_key}' to ROW access"
                    )
                    continue
                key = path_key.this
                if not exp.SAFE_IDENTIFIER_RE.match(key):
                    key = f'"{key}"'
                segments.append(f".{key}")

            expr = "".join(segments)

            return f"{this}{expr}"

        def groupconcat_sql(self, expression: exp.GroupConcat) -> str:
            return self.func(
                "ARRAY_JOIN",
                self.func("ARRAY_AGG", expression.this),
                expression.args.get("separator"),
            )
