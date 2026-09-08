module tsvc_2_s3110_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s3110_fp64(aa, bb, LEN_2D) bind(C, name="tsvc_2_s3110_fp64")
    ! Arguments: aa is input 2D array flattened row-major (size LEN_2D*LEN_2D)
    ! bb is output array (at least one element) where checksum is stored.
    integer(c_int64_t), value :: LEN_2D
    real(c_double), intent(in) :: aa(*)
    real(c_double), intent(out) :: bb(*)
    integer(c_int64_t) :: i, j, idx
    real(c_double) :: v, chksum
    real(c_double) :: global_max
    integer(c_int64_t) :: global_x, global_y
    real(c_double) :: local_max
    integer(c_int64_t) :: local_x, local_y

    ! Initialize global maximum with first element (aa[0] in C corresponds to aa(1) in Fortran)
    global_max = aa(1)
    global_x = 0_c_int64_t
    global_y = 0_c_int64_t

    ! Parallel region: each thread finds a local maximum and its position.
    !$omp parallel private(i, j, idx, v, local_max, local_x, local_y) default(none) shared(aa, LEN_2D, global_max, global_x, global_y)
      local_max = aa(1)
      local_x = 0_c_int64_t
      local_y = 0_c_int64_t
      !$omp do schedule(static) collapse(2)
      do i = 0_c_int64_t, LEN_2D - 1_c_int64_t
        do j = 0_c_int64_t, LEN_2D - 1_c_int64_t
          idx = i * LEN_2D + j
          v = aa(idx + 1)
          if (v > local_max) then
            local_max = v
            local_x = i
            local_y = j
          end if
        end do
      end do
      !$omp end do
      ! Update global maximum in a critical section.
      !$omp critical
        if (local_max > global_max) then
          global_max = local_max
          global_x = local_x
          global_y = local_y
        end if
      !$omp end critical
    !$omp end parallel

    chksum = global_max + dble(global_x) + dble(global_y)
    bb(1) = chksum
  end subroutine tsvc_2_s3110_fp64
end module tsvc_2_s3110_mod
