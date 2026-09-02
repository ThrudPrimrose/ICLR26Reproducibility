module tsvc_2_s1232_mod
  use iso_c_binding
  implicit none
contains

  subroutine tsvc_2_s1232_fp64(aa, bb, cc, LEN_2D, VLEN, workspace, workspace_bytes) bind(C, name="tsvc_2_s1232_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t), value :: VLEN
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    type(c_ptr), value :: aa
    type(c_ptr), value :: bb
    type(c_ptr), value :: cc
    real(c_double), pointer :: a_arr(:)
    real(c_double), pointer :: b_arr(:)
    real(c_double), pointer :: c_arr(:)
    integer(c_int64_t) :: i, j, i_start, idx

    call c_f_pointer(aa, a_arr, [LEN_2D*LEN_2D])
    call c_f_pointer(bb, b_arr, [LEN_2D*LEN_2D])
    call c_f_pointer(cc, c_arr, [LEN_2D*LEN_2D])

    ! Perform element‑wise addition for rows i >= j*VLEN (as defined by the benchmark)
    !$omp parallel do default(none) shared(a_arr, b_arr, c_arr, LEN_2D, VLEN) private(i, j, i_start, idx) schedule(static)
    do j = 0, LEN_2D-1
      i_start = j * VLEN
      if (i_start < LEN_2D) then
        !$omp simd
        do i = i_start, LEN_2D-1
          idx = i * LEN_2D + j
          a_arr(idx+1) = b_arr(idx+1) + c_arr(idx+1)
        end do
      end if
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s1232_fp64

end module tsvc_2_s1232_mod