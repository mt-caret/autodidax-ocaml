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
  type interpreter
  type global_data

  val level : int
  val global_data : global_data
end

module Abstract_value = struct
  type t =
    | Shaped_array of { dims : int array }
    | Concrete_array of Tensor.t
end

module type Tracer = sig
  type t
  type interpreter
  type value

  val interpreter : interpreter
  val aval : t -> Abstract_value.t
  val full_lower : t -> value
end

module type Interpreter0 = sig
  type t
  type packed_tracer

  module Tracer : Tracer with type interpreter = t
  module Level : Interpreter_level with type interpreter = t

  val tracer_witness : (t * Tracer.t) Type_equal.Id.t
  val pure : Tensor.t -> Tracer.t
  val lift : packed_tracer -> Tracer.t

  val process_primitive
    :  Primitive.t
    -> Tracer.t Nonempty_list.t
    -> Tracer.t Nonempty_list.t
end

module rec Interpreter : sig
  type 'tracer t =
    (module Interpreter0
       with type Tracer.t = 'tracer
        and type Tracer.value = Value.t
        and type packed_tracer = Packed_tracer.t)
end = struct
  type 'tracer t =
    (module Interpreter0
       with type Tracer.t = 'tracer
        and type Tracer.value = Value.t
        and type packed_tracer = Packed_tracer.t)
end

and Packed_interpreter : sig
  type t = T : 'tracer Interpreter.t -> t
end = struct
  type t = T : 'tracer Interpreter.t -> t
end

and Packed_tracer : sig
  type t =
    | T :
        { tracer : 'tracer
        ; interpreter : 'tracer Interpreter.t
        }
        -> t
end = struct
  type t =
    | T :
        { tracer : 'tracer
        ; interpreter : 'tracer Interpreter.t
        }
        -> t
end

and Value : sig
  type t =
    | Tensor of Tensor.t
    | Tracer of Packed_tracer.t

  val find_top_interpreter : t Nonempty_list.t -> Packed_interpreter.t option
  val full_lower : t -> t
  val full_raise : 'tracer Interpreter.t -> t -> 'tracer
end = struct
  type t =
    | Tensor of Tensor.t
    | Tracer of Packed_tracer.t

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

  let full_raise (type tracer) ((module Interpreter) : tracer Interpreter.t) t : tracer =
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

let bind prim values =
  let (T interpreter) =
    Value.find_top_interpreter values |> Option.value_exn (* FIXME *)
  in
  let tracers =
    Nonempty_list.map values ~f:(fun value -> Value.full_raise interpreter value)
  in
  let (module Interpreter) = interpreter in
  Interpreter.process_primitive prim tracers
  |> Nonempty_list.map ~f:Interpreter.Tracer.full_lower
;;
