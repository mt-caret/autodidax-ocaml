open! Core

module Primitive = struct
  type t =
    | Sin
    | Cos
    | Neg
    | Add
    | Sub
    | Mul
  [@@deriving sexp]
end

module type Interpreter_level = sig
  type global_data

  val level : int
  val global_data : global_data
end

module Abstract_value = struct
  type t =
    | Shaped_array of { dims : int array }
    | Concrete_array of Tensor.t

  let dims = function
    | Shaped_array { dims } -> dims
    | Concrete_array t -> Tensor.dims t
  ;;
end

module type Tracer = sig
  type t
  type value

  val aval : t -> Abstract_value.t
  val full_lower : t -> value
end

module type Interpreter0 = sig
  type packed_tracer

  module Tracer : Tracer
  module Level : Interpreter_level

  val tracer_witness : Tracer.t Type_equal.Id.t
  val pure : Tensor.t -> Tracer.t
  val lift : packed_tracer -> Tracer.t

  val process_primitive
    :  Primitive.t
    -> Tracer.t Nonempty_list.t
    -> Tracer.t Nonempty_list.t
end

module rec Interpreter : sig
  type ('tracer, 'global_data) t =
    (module Interpreter0
       with type Tracer.t = 'tracer
        and type Tracer.value = Value.t
        and type Level.global_data = 'global_data
        and type packed_tracer = Packed_tracer.t)
end = struct
  type ('tracer, 'global_data) t =
    (module Interpreter0
       with type Tracer.t = 'tracer
        and type Tracer.value = Value.t
        and type Level.global_data = 'global_data
        and type packed_tracer = Packed_tracer.t)
end

and Packed_interpreter : sig
  type t = T : ('tracer, 'global_data) Interpreter.t -> t
end = struct
  type t = T : ('tracer, 'global_data) Interpreter.t -> t
end

and Packed_tracer : sig
  type t =
    | T :
        { tracer : 'tracer
        ; interpreter : ('tracer, 'global_data) Interpreter.t
        }
        -> t
  [@@deriving sexp_of]
end = struct
  type t =
    | T :
        { tracer : 'tracer
        ; interpreter : (('tracer, 'global_data) Interpreter.t[@sexp.opaque])
        }
        -> t
  [@@deriving sexp_of]
end

and Value : sig
  type t =
    | Tensor of Tensor.t
    | Tracer of Packed_tracer.t
  [@@deriving sexp_of]

  val of_float : float -> t
  val get_aval : t -> Abstract_value.t
  val find_top_interpreter : t Nonempty_list.t -> Packed_interpreter.t option
  val full_lower : t -> t
  val full_raise : ('tracer, _) Interpreter.t -> t -> 'tracer
end = struct
  type t =
    | Tensor of Tensor.t
    | Tracer of Packed_tracer.t
  [@@deriving sexp_of]

  let of_float f = Tensor (Tensor.of_float f)

  let get_aval t =
    match t with
    | Tensor t -> Abstract_value.Concrete_array t
    | Tracer (T { tracer; interpreter = (module Interpreter) }) ->
      Interpreter.Tracer.aval tracer
  ;;

  let find_top_interpreter values =
    Nonempty_list.filter_map values ~f:(function
      | Tensor _ -> None
      | Tracer packed_tracer -> Some packed_tracer)
    |> List.max_elt
         ~compare:
           (Comparable.lift
              Int.compare
              ~f:
                (fun
                  (Packed_tracer.T { tracer = _; interpreter = (module Interpreter) }) ->
                Interpreter.Level.level))
    |> Option.map ~f:(fun (Packed_tracer.T { tracer = _; interpreter }) ->
      Packed_interpreter.T interpreter)
  ;;

  let full_lower = function
    | Tensor t -> Tensor t
    | Tracer (T { tracer; interpreter = (module Interpreter) }) ->
      Interpreter.Tracer.full_lower tracer
  ;;

  let full_raise
    (type tracer global_data)
    ((module Interpreter) : (tracer, global_data) Interpreter.t)
    t
    : tracer
    =
    match t with
    | Tensor t -> Interpreter.pure t
    | Tracer (T { tracer; interpreter = (module Interpreter_to_lift) } as packed_tracer)
      ->
      (match
         Ordering.of_int (compare Interpreter.Level.level Interpreter_to_lift.Level.level)
       with
       | Equal ->
         let T =
           Type_equal.Id.same_witness_exn
             Interpreter.tracer_witness
             Interpreter_to_lift.tracer_witness
         in
         tracer
       | Greater -> Interpreter.lift packed_tracer
       | Less ->
         raise_s
           [%message "Cannot lift a higher-level tracer to a lower-level interpreter"])
  ;;
end

let interpreter_stack : Packed_interpreter.t list ref = ref []

let new_interpreter
  (type tracer global_data)
  ~(create_interpreter : level:int -> global_data -> (tracer, global_data) Interpreter.t)
  ~global_data
  ~f
  =
  let level = List.length !interpreter_stack in
  let interpreter = create_interpreter ~level global_data in
  interpreter_stack := Packed_interpreter.T interpreter :: !interpreter_stack;
  protect
    ~f:(fun () -> f interpreter)
    ~finally:(fun () -> interpreter_stack := List.tl_exn !interpreter_stack)
;;

let bind prim values =
  let (T interpreter) =
    Value.find_top_interpreter values
    |> Option.value ~default:(List.last_exn !interpreter_stack)
  in
  let tracers =
    Nonempty_list.map values ~f:(fun value -> Value.full_raise interpreter value)
  in
  let (module Interpreter) = interpreter in
  Interpreter.process_primitive prim tracers
  |> Nonempty_list.map ~f:Interpreter.Tracer.full_lower
;;

let bind1 prim values =
  match bind prim values with
  | [ t ] -> t
  | _ -> raise_s [%message "Expected a single result" (prim : Primitive.t)]
;;

let ( + ) t1 t2 = bind1 Primitive.Add [ t1; t2 ]
let ( - ) t1 t2 = bind1 Primitive.Sub [ t1; t2 ]
let ( * ) t1 t2 = bind1 Primitive.Mul [ t1; t2 ]
let ( ~- ) t = bind1 Primitive.Neg [ t ]
let sin t = bind1 Primitive.Sin [ t ]
let cos t = bind1 Primitive.Cos [ t ]

let eval_interpreter ~level : (Tensor.t, unit) Interpreter.t =
  (module struct
    type packed_tracer = Packed_tracer.t

    module Tracer = struct
      type t = Tensor.t
      type value = Value.t

      let aval t = Abstract_value.Concrete_array t
      let full_lower t = Value.Tensor t
    end

    module Level = struct
      type global_data = unit

      let level = level
      let global_data = ()
    end

    let tracer_witness = Type_equal.Id.create ~name:"tensor" [%sexp_of: Tensor.t]
    let pure = Fn.id

    let lift packed_tracer =
      raise_s
        [%message "Cannot lift in eval interpreter" (packed_tracer : Packed_tracer.t)]
    ;;

    let process_primitive (prim : Primitive.t) (values : Tensor.t Nonempty_list.t)
      : Tensor.t Nonempty_list.t
      =
      match prim, values with
      | Sin, [ t ] -> [ Tensor.sin t ]
      | Cos, [ t ] -> [ Tensor.cos t ]
      | Neg, [ t ] -> [ Tensor.( ~- ) t ]
      | Add, [ t1; t2 ] -> [ Tensor.( + ) t1 t2 ]
      | Sub, [ t1; t2 ] -> [ Tensor.( - ) t1 t2 ]
      | Mul, [ t1; t2 ] -> [ Tensor.( * ) t1 t2 ]
      | _ ->
        raise_s
          [%message
            "unexpected evaluation" (prim : Primitive.t) (values : _ Nonempty_list.t)]
    ;;
  end)
;;

let () = interpreter_stack := [ Packed_interpreter.T (eval_interpreter ~level:0) ]

let%expect_test "eval" =
  let f x =
    let y = sin x * Value.of_float 2. in
    let z = -y + x in
    z
  in
  Value.of_float 3. |> f |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor 2.7177599838802657) |}]
;;

module Jvp_tracer = struct
  type t =
    { primal : Value.t
    ; tangent : Value.t
    }
  [@@deriving sexp_of]
end

let jvp_interpreter ~level () : (Jvp_tracer.t, unit) Interpreter.t =
  let rec interpreter : (Jvp_tracer.t, unit) Interpreter.t lazy_t =
    lazy
      (module struct
        type packed_tracer = Packed_tracer.t

        module Tracer = struct
          type t = Jvp_tracer.t
          type value = Value.t

          let aval { Jvp_tracer.primal; tangent = _ } = Value.get_aval primal

          let full_lower t =
            Value.Tracer (T { tracer = t; interpreter = force interpreter })
          ;;
        end

        module Level = struct
          type global_data = unit

          let level = level
          let global_data = ()
        end

        let tracer_witness =
          Type_equal.Id.create ~name:"jvp_tracer" [%sexp_of: Jvp_tracer.t]
        ;;

        let value_to_tracer value =
          { Jvp_tracer.primal = value
          ; tangent =
              Tensor (Tensor.zeros ~dims:(Value.get_aval value |> Abstract_value.dims))
          }
        ;;

        let pure tensor = value_to_tracer (Tensor tensor)
        let lift packed_tracer = value_to_tracer (Tracer packed_tracer)

        let process_primitive (prim : Primitive.t) (values : Jvp_tracer.t Nonempty_list.t)
          : Jvp_tracer.t Nonempty_list.t
          =
          match prim, values with
          | Sin, [ { primal = x; tangent = dx } ] ->
            [ { primal = sin x; tangent = cos x * dx } ]
          | Cos, [ { primal = x; tangent = dx } ] ->
            [ { primal = cos x; tangent = -sin x * dx } ]
          | Neg, [ { primal = x; tangent = dx } ] -> [ { primal = -x; tangent = -dx } ]
          | Add, [ { primal = x1; tangent = dx1 }; { primal = x2; tangent = dx2 } ] ->
            [ { primal = x1 + x2; tangent = dx1 + dx2 } ]
          | Sub, [ { primal = x1; tangent = dx1 }; { primal = x2; tangent = dx2 } ] ->
            [ { primal = x1 - x2; tangent = dx1 - dx2 } ]
          | Mul, [ { primal = x1; tangent = dx1 }; { primal = x2; tangent = dx2 } ] ->
            [ { primal = x1 * x2; tangent = (x1 * dx2) + (dx1 * x2) } ]
          | _ ->
            raise_s
              [%message
                "unexpected evaluation" (prim : Primitive.t) (values : _ Nonempty_list.t)]
        ;;
      end)
  in
  force interpreter
;;

let jvp ~f primals tangets =
  new_interpreter
    ~create_interpreter:jvp_interpreter
    ~global_data:()
    ~f:(fun interpreter ->
      let tracers_in =
        Nonempty_list.zip_exn primals tangets
        |> Nonempty_list.map ~f:(fun (primal, tangent) ->
          Value.Tracer (T { tracer = { Jvp_tracer.primal; tangent }; interpreter }))
      in
      let out = f tracers_in in
      let { Jvp_tracer.primal; tangent } = Value.full_raise interpreter out in
      primal, tangent)
;;

let jvp1 ~f primal tangent =
  jvp
    ~f:(function
      | Nonempty_list.[ x ] -> f x
      | _ -> assert false)
    [ primal ]
    [ tangent ]
;;

let%expect_test "jvp" =
  let x = Value.of_float 3. in
  let dx = Value.of_float 1. in
  jvp1 ~f:sin x dx |> [%sexp_of: Value.t * Value.t] |> print_s;
  [%expect {| ((Tensor 0.14112000805986721) (Tensor -0.98999249660044542)) |}];
  cos x |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor -0.98999249660044542) |}];
  let f x =
    let y = sin x * Value.of_float 2. in
    let z = -y + x in
    z
  in
  let x, dx = Value.of_float 3., Value.of_float 1. in
  let y, dy = jvp1 ~f x dx in
  [%sexp_of: Value.t * Value.t] (y, dy) |> print_s;
  [%expect {| ((Tensor 2.7177599838802657) (Tensor 2.9799849932008908)) |}];
  let deriv ~f x = jvp1 ~f x (Value.of_float 1.) |> snd in
  deriv ~f:sin x |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor -0.98999249660044542) |}];
  deriv ~f:(deriv ~f:sin) x |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor -0.14112000805986721) |}];
  deriv ~f:(deriv ~f:(deriv ~f:sin)) x |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor 0.98999249660044542) |}];
  deriv ~f:(deriv ~f:(deriv ~f:(deriv ~f:sin))) x |> [%sexp_of: Value.t] |> print_s;
  [%expect {| (Tensor 0.14112000805986721) |}]
;;
