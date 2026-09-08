subroutine segment_reduce_ragged_fp64(val, row_ptr, w, out, NSEG, workspace, workspace_size) bind(C)
    use iso_c_binding
    implicit none
    ! Arguments
    integer(c_int64_t), value, intent(in) :: NSEG
    type(c_ptr), value, intent(in) :: workspace
    real(c_double), intent(in) :: val(*)
    integer(c_int64_t), intent(in) :: row_ptr(*)
    integer(c_int64_t), value, intent(in) :: workspace_size
    real(c_double), intent(in) :: w(*)
    real(c_double), intent(out) :: out(*)
    ! Local variables
    integer(c_int64_t) :: s
    integer(c_int64_t) :: start, finish, e
    integer(c_int64_t) :: offset
    real(c_double) :: acc
    offset = 1_c_int64_t - row_ptr(1)
    ! Parallelize outer loop over segments using OpenMP
    !$omp parallel do default(none) shared(row_ptr, val, w, out, NSEG, offset) private(s, start, finish, e, acc) schedule(dynamic)
    do s = 1, NSEG
        start = row_ptr(s) + offset
        finish = row_ptr(s+1) + offset - 1_c_int64_t
        acc = 0.0_c_double
        if (start <= finish) then
            do e = start, finish
                acc = acc + val(e) * w(e)
            end do
        end if
        out(s) = acc
    end do
    !$omp end parallel do
end subroutine segment_reduce_ragged_fp64
